#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* getopt */
#include <limits.h>
#include <time.h>

#define MAX_ITERA 100000
#define BlocksPerGrid ((COUNT - 1) / ThreadsPerBlock + 1)
#define ThreadsPerBlock 1024
#define RND() ((double)rand() / RAND_MAX)
#define min(x, y) (x) < (y) ? (x) : (y)
#define COUNT 4096

typedef struct tag_particle_Coal
{
    double *position;
    double *velocity;
    double *fitness;
    double *pbest_pos;
    double *pbest_fit;
} particle_Coal;

typedef struct tag_particle
{
    double *position;
    double *velocity;
    double *fitness;
} particle; // this is only for gbest

typedef struct tag_arguments
{
    int max_iter;
    int particle_cnt;
    int threads_per_block;
    int blocks_per_grid;
    int dimensions;
    int verbose;
} arguments;

double w, c1, c2;
double max_v;
double max_pos, min_pos;
unsigned int particle_cnt;

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int parseArgs(arguments *args, int argc, char **argv)
{
    int cmd_opt = 0;

    while (1)
    {
        cmd_opt = getopt(argc, argv, "v:m:c:t:d:");
        if (cmd_opt == -1)
        {
            break;
        }
        switch (cmd_opt)
        {
        case 'm':
            args->max_iter = atoi(optarg);
            break;
        case 'c':
            args->particle_cnt = atoi(optarg);
            break;
        case 't':
            args->threads_per_block = atoi(optarg);
            break;
        case 'v':
            args->verbose = atoi(optarg);
            break;
        case 'd':
            args->dimensions = atoi(optarg);
            break;
        /* Error handle: Mainly missing arg or illegal option */
        case '?':
            fprintf(stderr, "Illegal option %c \n", (char)optopt);
            break;
        default:
            fprintf(stderr, "Not supported option\n");
            break;
        }
    }

    if (args->threads_per_block && args->particle_cnt)
        args->blocks_per_grid = ((args->particle_cnt - 1) / args->threads_per_block + 1);
    if (args->verbose)
    {
        printf("max_iter is %d \n", args->max_iter);
        printf("particle_cnt is %d \n", args->particle_cnt);
        printf("threads_per_block is %d \n", args->threads_per_block);
        printf("blocks_per_grid is %d \n", args->blocks_per_grid);
        printf("=============================\n");
    }
    return 1;
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__host__ __device__ double fit(double *x, int d);
__global__ void findBestInOneBlock(double *aux, double *aux_pos, int d, double *gbest_position_d, double *gbest_fitness_d);
__global__ void move(double *position_d, double *velocity_d, double *fitness_d,
                     double *pbest_pos_d, double *pbest_fit_d, int *lock, volatile double *aux, volatile double *aux_pos,
                     int d, double *gbest_position_d, double *gbest_fitness_d, volatile double *privateBestPosQueue_d, int iter);
void ParticleInitCoal(particle_Coal *p, int d, particle *gbest);
void ParticleFreeCoal(particle_Coal *p);
void ParticleInitGbest(particle *gbest_d, int d);
void ParticleFreeGbest(particle *gbest_d);

// cuda constant memory
__constant__ double w_d;
__constant__ double c1_d;
__constant__ double c2_d;
__constant__ double max_pos_d;
__constant__ double min_pos_d;
__constant__ double max_v_d;
__constant__ int max_iter_d;
__constant__ int particle_cnt_d;
__constant__ int num_dimensions; // might need it?
__constant__ int tile_size;
__constant__ int tile_size2;

__global__ void findBestInOneBlock(double *aux, double *aux_pos, int d, double *gbest_position_d, double *gbest_fitness_d)
{
    int idx = threadIdx.x;
    extern __shared__ double privateBestShit[];
    double *privateBest = &privateBestShit[0];
    double *privateBestIndex = &privateBestShit[blockDim.x];
    if (idx < tile_size2)
    {
        privateBest[idx] = aux[idx];
        privateBestIndex[idx] = idx;
    }
    else
    {
        privateBest[idx] = -1.0;
    }
    __syncthreads();

    if (tile_size2 % 2 != 0 && idx == tile_size2 - 2)
    {
        if (privateBest[tile_size2 - 2] < privateBest[tile_size2 - 1])
        {
            privateBest[tile_size2 - 2] = privateBest[tile_size2 - 1];
            privateBestIndex[tile_size2 - 2] = tile_size2 - 1;
        }
    }
    __syncthreads();

    // Reduction loop
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (idx < stride && idx + stride < tile_size2)
        {
            if (privateBest[idx] < privateBest[idx + stride])
            {
                privateBest[idx] = privateBest[idx + stride];
                privateBestIndex[idx] = idx + stride;
            }
        }
        __syncthreads();
    }

    // Update global best fitness and position
    if (idx == 0)
    {
        if (privateBest[0] > gbest_fitness_d[0])
        {
            gbest_fitness_d[0] = privateBest[0];
            int largest_idx = privateBestIndex[0];
            for (int j = 0; j < d; j++)
            {
                gbest_position_d[j] = aux_pos[largest_idx * d + j];
            }
            __threadfence_block();
        }
    }
}

__global__ void move(double *position_d, double *velocity_d, double *fitness_d,
                     double *pbest_pos_d, double *pbest_fit_d, int *lock, volatile double *aux, volatile double *aux_pos,
                     int d, double *gbest_position_d, double *gbest_fitness_d, volatile double *privateBestPosQueue_d, int iter)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    extern __shared__ double sharedMemory[];
    double *privateBestQueue = (double *)sharedMemory;
    // double *privateBestPosQueue = (double *)&sharedMemory[tile_size];
    __shared__ unsigned int queue_num;
    if (threadIdx.x == 0)
    {
        queue_num = 0;
        privateBestQueue[0] = INT_MIN;
    }
    __syncthreads();
    curandState state1, state2;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state1);
    curand_init((unsigned long long)clock() + idx, 0, 0, &state2);
    double fitness = fitness_d[idx];
    if (idx < particle_cnt_d)
    {
        for (int j = 0; j < d; j++)
        {
            double v = velocity_d[idx * d + j];
            double pos = position_d[idx * d + j];
            double ppos = pbest_pos_d[idx * d + j];
            velocity_d[idx * d + j] = w_d * v + c1_d * curand_uniform_double(&state1) * (ppos - pos) + c2_d * curand_uniform_double(&state2) * (gbest_position_d[j] - pos);
            if (velocity_d[idx * d + j] < -max_v_d)
                velocity_d[idx * d + j] = -max_v_d;
            else if (velocity_d[idx * d + j] > max_v_d)
                velocity_d[idx * d + j] = max_v_d;
            position_d[idx * d + j] = pos + v;
            if (position_d[idx * d + j] > max_pos_d)
                position_d[idx * d + j] = max_pos_d;
            else if (position_d[idx * d + j] < min_pos_d)
                position_d[idx * d + j] = min_pos_d;
        }
        fitness = fit(position_d + idx * d, d);
        if (fitness > pbest_fit_d[idx])
        {
            for (int j = 0; j < d; j++)
            {
                pbest_pos_d[idx * d + j] = position_d[idx * d + j];
            }
            pbest_fit_d[idx] = fitness;
        }
    }
    __syncthreads();
    if (fitness > gbest_fitness_d[0])
    {
        unsigned const my_index = atomicAdd(&queue_num, 1) + 1;
        privateBestQueue[my_index] = fitness;

        for (int j = 0; j < d; j++)
        {
            privateBestPosQueue_d[blockIdx.x * blockDim.x * d + my_index * d + j] = position_d[idx * d + j];
        }
    }
    __syncthreads();
    if (idx < particle_cnt_d)
    {
        if (tidx == 0)
        {
            int maxx = -1;
            aux[blockIdx.x] = INT_MIN;
            aux_pos[blockIdx.x] = INT_MIN;
            if (queue_num)
            {
                for (int j = 1; j <= queue_num; j++)
                {
                    if (privateBestQueue[j] >= privateBestQueue[0])
                    {
                        maxx = j;
                        privateBestQueue[0] = privateBestQueue[j];
                    }
                }
                if (maxx != -1)
                {
                    aux[blockIdx.x] = privateBestQueue[0];
                    for (int k = 0; k < d; k++)
                    {
                        aux_pos[blockIdx.x * d + k] = privateBestPosQueue_d[blockIdx.x * blockDim.x * d + (maxx)*d + k];
                    }
                }
            }
        }
    }
}

__host__ __device__ double fit(double *x, int d)
{
    double summation = 0.0;
    for (int i = 0; i < d; i++)
    {
        summation += fabs(8000.0 + x[i] * (-10000.0 + x[i] * (-0.8 + x[i])));
    }
    return summation;
}

int main(int argc, char **argv)
{
    arguments args = {};
    args.max_iter = 10000;
    args.particle_cnt = 4096;
    args.threads_per_block = 128;
    args.dimensions = 64;
    args.blocks_per_grid = ((args.particle_cnt - 1) / args.threads_per_block + 1);
    args.verbose = 1;
    int retError = parseArgs(&args, argc, argv);
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float exe_time;
    clock_t begin_app = clock();
    clock_t begin_init = begin_app;
    particle_Coal *p;
    particle *gbest;

    double *position_d;
    double *velocity_d;
    double *fitness_d;
    double *pbest_pos_d;
    double *pbest_fit_d;

    double *gbest_position_d;
    double *gbest_velocity_d;
    double *gbest_fitness_d;

    double *privateBestPosQueue_d;

    double *aux, *aux_pos;
    int *lock_d;

    int block_size = args.blocks_per_grid;

    // Set parameters
    min_pos = -100.0, max_pos = +100.0;
    w = 1, c1 = 2.0, c2 = 2.0;
    particle_cnt = args.particle_cnt;
    max_v = (max_pos - min_pos) * 1.0;

    p = (particle_Coal *)malloc(sizeof(particle_Coal));
    gbest = (particle *)malloc(sizeof(particle));

    ParticleInitGbest(gbest, args.dimensions);
    ParticleInitCoal(p, args.dimensions, gbest);

    printf("Allocating device memory\n");
    HANDLE_ERROR(cudaMalloc((void **)&position_d, sizeof(double) * particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&privateBestPosQueue_d, sizeof(double) * particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, sizeof(double) * particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, sizeof(double) * particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(double) * particle_cnt));

    // HANDLE_ERROR(cudaMalloc((void **)&gbest_d, sizeof(particle)));// need to change this.. multiple lines
    HANDLE_ERROR(cudaMalloc((void **)&gbest_position_d, sizeof(double) * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_velocity_d, sizeof(double) * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_fitness_d, sizeof(double) * 3));

    HANDLE_ERROR(cudaMalloc((void **)&lock_d, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(double) * args.blocks_per_grid * args.dimensions));

    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, sizeof(double) * particle_cnt * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, sizeof(double) * particle_cnt * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, sizeof(double) * particle_cnt * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(gbest_position_d, gbest->position, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_velocity_d, gbest->velocity, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_fitness_d, gbest->fitness, sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpyToSymbol(w_d, &w, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c1_d, &c1, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c2_d, &c2, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_pos_d, &max_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(min_pos_d, &min_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_v_d, &max_v, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_iter_d, &args.max_iter, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(particle_cnt_d, &args.particle_cnt, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(num_dimensions, &args.dimensions, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size2, &block_size, sizeof(int)));
    HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));
    clock_t end_init = clock();
    clock_t begin_exe = end_init;
    HANDLE_ERROR(cudaEventRecord(start));
    for (unsigned int i = 0; i < args.max_iter; i++)
    {
        move<<<args.blocks_per_grid, args.threads_per_block, sizeof(double) * (args.threads_per_block + 1)>>>(position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d,
                                                                                                              lock_d, aux, aux_pos, args.dimensions, gbest_position_d, gbest_fitness_d, privateBestPosQueue_d, i);
        cudaDeviceSynchronize();
        findBestInOneBlock<<<1, block_size, sizeof(double) * block_size * 2>>>(aux, aux_pos, args.dimensions, gbest_position_d, gbest_fitness_d);
        cudaDeviceSynchronize();
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaMemcpy(gbest->position, gbest_position_d, sizeof(double) * args.dimensions, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(gbest->fitness, gbest_fitness_d, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    clock_t end_exe = clock();
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&exe_time, start, stop));

    printf("best result: %lf\n", gbest->fitness[0]);
    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    printf("[Cuda Exec time]: %f (sec)\n", exe_time / 1000);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    ParticleFreeCoal(p);
    ParticleFreeGbest(gbest);
    cudaFree(position_d);
    cudaFree(velocity_d);
    cudaFree(fitness_d);
    cudaFree(pbest_pos_d);
    cudaFree(pbest_fit_d);
    cudaFree(lock_d);
    return 0;
}

void ParticleInitCoal(particle_Coal *p, int d, particle *gbest)
{
    unsigned int i;
    unsigned int j;
    const double pos_range = max_pos - min_pos;
    srand((unsigned)time(NULL));
    p->position = (double *)malloc(sizeof(double) * particle_cnt * d);
    p->velocity = (double *)malloc(sizeof(double) * particle_cnt * d);
    p->fitness = (double *)malloc(sizeof(double) * particle_cnt);
    p->pbest_pos = (double *)malloc(sizeof(double) * particle_cnt * d);
    p->pbest_fit = (double *)malloc(sizeof(double) * particle_cnt);

    for (i = 0; i < particle_cnt; i++)
    {
        for (j = 0; j < d; j++)
        {

            p->pbest_pos[i * d + j] = p->position[i * d + j] = RND() * pos_range + min_pos; // removed
            // p->pbest_pos[i*d+j] = p->position[i*d +j] = 0.0;  // added

            // printf("position value : %f\n", p->position[i*d +j]);
            // printf("RND() in init %f\n", RND());
            p->velocity[i * d + j] = RND() * max_v;
        }
        p->pbest_fit[i] = p->fitness[i] = fit(p->position + i * d, d); // removed
        // p->pbest_fit[i] = p->fitness[i] = 0.0;   // added

        if (i == 0 || p->pbest_fit[i] > gbest->fitness[0])
        {
            for (j = 0; j < d; j++)
            {
                gbest->position[j] = p->position[i * d + j];
                gbest->velocity[j] = p->velocity[i * d + j];
            }
            gbest->fitness[0] = p->fitness[i];
        }
    }
}

void ParticleFreeCoal(particle_Coal *p)
{

    free(p->position);
    free(p->velocity);
    free(p->fitness);
    free(p->pbest_pos);
    free(p->pbest_fit);
    free(p);
}

void ParticleInitGbest(particle *gbest_d, int d)
{
    gbest_d->position = (double *)malloc(sizeof(double) * d);
    gbest_d->velocity = (double *)malloc(sizeof(double) * d);
    gbest_d->fitness = (double *)malloc(sizeof(double) * 3);

    gbest_d->fitness[0] = -10000000.0; // Should not matter, will change in ParticleInitCoal, should be small
}

void ParticleFreeGbest(particle *gbest_d)
{

    free(gbest_d->position);
    free(gbest_d->velocity);
    free(gbest_d->fitness);
    free(gbest_d);
}
