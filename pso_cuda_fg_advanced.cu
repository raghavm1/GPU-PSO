#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <float.h>

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
    double fitness;
    double *pbest_pos;
    double pbest_fit;
    double *g_fitness;
} particle;

typedef struct tag_arguments
{
    int max_iter;
    int particle_cnt;
    int threads_per_block;
    int blocks_per_grid;
    int verbose;
    int dimensions;
} arguments;

double w, c1, c2;
double max_v;
double max_pos, min_pos;
unsigned int particle_cnt;

particle *gbest;

void initialize_gbest(particle *gbest, int dimensions)
{
    gbest->position = (double *)malloc(sizeof(double) * dimensions);
    gbest->velocity = NULL;
    gbest->pbest_pos = (double *)malloc(sizeof(double) * dimensions);

    gbest->g_fitness = (double *)malloc(sizeof(double) * 1);
    gbest->g_fitness[0] = -DBL_MAX;
    gbest->pbest_fit = -DBL_MAX;

    for (int i = 0; i < dimensions; i++)
    {
        gbest->position[i] = 0.0;
        gbest->pbest_pos[i] = 0.0;
    }
}

void free_gbest()
{
    free(gbest->position);
    free(gbest->pbest_pos);
}

void ParticleInitCoal(particle_Coal *p);
void ParticleInit(particle *p);

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

int pargeArgs(arguments *args, int argc, char **argv)
{
    int cmd_opt = 0;
    while (1)
    {
        cmd_opt = getopt(argc, argv, "v:m:c:t:b:d:");
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
        case 'b':
            args->blocks_per_grid = atoi(optarg);
            break;
        case 'd': // dimensions
            if (optarg == NULL)
            {
                args->dimensions = 1;
            }
            else
                args->dimensions = atoi(optarg);
            break;
        case 'v':
            args->verbose = atoi(optarg);
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
    if (args->verbose)
    {
        printf("max_iter is %d \n", args->max_iter);
        printf("particle_cnt is %d \n", args->particle_cnt);
        printf("number of dimensions are %d \n", args->dimensions);
        printf("threads_per_block is %d \n", args->threads_per_block);
        printf("blocks_per_grid is %d \n", args->blocks_per_grid);
        printf("=============================\n");
    }
    return 1;
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// cuda constant memory
__constant__ double w_d;
__constant__ double c1_d;
__constant__ double c2_d;
__constant__ double max_pos_d;
__constant__ double min_pos_d;
__constant__ double max_v_d;
__constant__ int max_iter_d;
__constant__ int particle_cnt_d;
__constant__ int tile_size;
__constant__ int tile_size2;

void calculate_launch_params(int particle_count, int dimensions,
                             int threads_per_block, int num_blocks, int *dimensions_per_thread,
                             int *particles_per_block)
{

    *particles_per_block = (particle_count + num_blocks - 1) / num_blocks; // this way does ceil(particle_count/num_blocks)

    int num_dimensions_per_block = *particles_per_block * dimensions;

    *dimensions_per_thread = (num_dimensions_per_block + threads_per_block - 1) / threads_per_block; // similarly this also is ceil
}

__host__ __device__ double fit(double x)
{
    return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x)));
}

__global__ void updateParticles(double *position_d, double *velocity_d, double *fitness_d,
                                double *pbest_pos_d, double *pbest_fit_d,
                                double *best_fitness_buf_d, double *best_positions_buf_d,
                                int dim_d, double *gbest_position_d, double *gbest_fitness_d,
                                int particles_per_block, int dimensions_per_thread)
{
    extern __shared__ double shared_mem[];

    double *privateBestQueue = &shared_mem[0];
    double *privateBestQueueIndex = &shared_mem[particles_per_block];

    __shared__ unsigned int queue_num;
    if (threadIdx.x == 0)
    {
        queue_num = 0;
        privateBestQueue[0] = INT_MIN;
    }
    __syncthreads();

    int first_particle = blockIdx.x * particles_per_block;
    int thread_idx = threadIdx.x;

    int total_elements = particles_per_block * dim_d;
    int elements_per_thread = dimensions_per_thread;
    int start_element = thread_idx * elements_per_thread;
    int end_element = min(start_element + elements_per_thread, total_elements);

    for (int i = start_element; i < end_element; i++)
    {
        int particle_offset = i / dim_d;
        int dim_offset = i % dim_d;
        int global_particle_idx = first_particle + particle_offset;

        if (global_particle_idx < particle_cnt_d)
        {
            int global_idx = global_particle_idx * dim_d + dim_offset;

            curandState state;
            curand_init((unsigned long long)clock() + global_idx, 0, 0, &state);

            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);

            double v = w_d * velocity_d[global_idx] +
                       c1_d * r1 * (pbest_pos_d[global_idx] - position_d[global_idx]) +
                       c2_d * r2 * (gbest_position_d[dim_offset] - position_d[global_idx]);

            v = max(min(v, max_v_d), -max_v_d);

            double pos = position_d[global_idx] + v;
            pos = max(min(pos, max_pos_d), min_pos_d);

            position_d[global_idx] = pos;
            velocity_d[global_idx] = v;
        }
    }

    __syncthreads();

    double fitness = 0.0;
    if ((thread_idx + 1) * elements_per_thread % dim_d == 0)
    {
        start_element = thread_idx * elements_per_thread;
        int new_end_element = min(start_element + dim_d, total_elements); // this new_end_element is end elemt of entire particle

        for (int i = start_element; i < end_element; i++)
        {
            int particle_offset = i / dim_d; // Which particle in this block
            int dim_offset = i % dim_d;      // Which dimension of the particle
            int global_particle_idx = first_particle + particle_offset;
            int global_idx = global_particle_idx * dim_d + dim_offset;
            fitness += fit(position_d[global_idx]);
        }

        int particle_offset = start_element / dim_d;
        int global_particle_idx = first_particle + particle_offset;
        if (fitness > pbest_fit_d[global_particle_idx])
        {
            pbest_fit_d[global_particle_idx] = fitness;

            for (int i = start_element; i < new_end_element; i++)
            {
                int dim_offset = i % dim_d;
                int global_idx = global_particle_idx * dim_d + dim_offset;
                pbest_pos_d[global_idx] = position_d[global_idx];
            }
        }

        if (fitness > gbest_fitness_d[0])
        {
            unsigned const my_index = atomicAdd(&queue_num, 1) + 1;
            privateBestQueue[my_index] = fitness;
            privateBestQueueIndex[my_index] = global_particle_idx;
        }
    }

    __syncthreads();

    if (thread_idx == 0)
    {
        int maxx = -1;
        best_fitness_buf_d[blockIdx.x] = INT_MIN;
        best_positions_buf_d[blockIdx.x] = INT_MIN;
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
                best_fitness_buf_d[blockIdx.x] = privateBestQueue[0];
                for (int k = 0; k < dim_d; k++)
                {
                    int global_particle_idx = privateBestQueueIndex[maxx];
                    best_positions_buf_d[blockIdx.x * dim_d + k] = position_d[global_particle_idx + k];
                }
            }
        }
    }
}

__global__ void findGlobalBest(double *best_fitness_buf, double *best_positions_buf, int particle_count, int dim_d, double *gbest_fitness_d, double *gbest_position_d)
{
    extern __shared__ double shared_data[];
    double *s_fitness = &shared_data[0];
    double *s_indices = &shared_data[blockDim.x];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < particle_count)
    {
        s_fitness[tid] = best_fitness_buf[gid];
        s_indices[tid] = gid;
    }
    else
    {
        s_fitness[tid] = -INFINITY;
        s_indices[tid] = -1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && gid < particle_count)
        {
            if (s_fitness[tid] < s_fitness[tid + stride])
            {
                s_fitness[tid] = s_fitness[tid + stride];
                s_indices[tid] = s_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (s_fitness[0] > gbest_fitness_d[0])
        {
            int best_idx = (int)s_indices[0];
            if (best_idx >= 0)
            {
                gbest_fitness_d[0] = s_fitness[0];

                for (int d = 0; d < dim_d; d++)
                {
                    gbest_position_d[d] = best_positions_buf[best_idx * dim_d + d];
                }
            }
        }
    }
}

void ParticleInitCoal(particle_Coal *p, int dimensions)
{
    const double pos_range = max_pos - min_pos;
    srand((unsigned)time(NULL));

    p->position = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->velocity = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->fitness = (double *)malloc(sizeof(double) * particle_cnt);
    p->pbest_pos = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->pbest_fit = (double *)malloc(sizeof(double) * particle_cnt);

    for (int i = 0; i < particle_cnt; i++)
    {
        double fitness = 0.0;

        for (int d = 0; d < dimensions; d++)
        {
            int idx = i * dimensions + d;
            p->position[idx] = double(RND() * pos_range + min_pos);
            p->velocity[idx] = RND() * max_v;
            p->pbest_pos[idx] = double(p->position[idx]);
            fitness += fit(p->position[idx]);
        }

        p->fitness[i] = fitness;
        p->pbest_fit[i] = fitness;
        if (fitness > gbest->g_fitness[0])
        {
            gbest->g_fitness[0] = fitness;
            for (int d = 0; d < dimensions; d++)
            {
                gbest->position[d] = p->position[i * dimensions + d];
            }
        }
    }
}

int main(int argc, char **argv)
{
    arguments args = {};
    args.max_iter = 10000;
    args.particle_cnt = 4096;
    args.dimensions = 64;
    args.threads_per_block = 256;
    args.blocks_per_grid = 256;
    int retError = pargeArgs(&args, argc, argv);
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float exe_time;

    clock_t begin_app = clock();
    clock_t begin_init = begin_app;
    particle_Coal *p;
    p = (particle_Coal *)malloc(sizeof(particle_Coal));
    double *position_d;
    double *velocity_d;
    double *fitness_d;
    double *pbest_pos_d;
    double *pbest_fit_d;
    double *aux, *aux_pos;
    double *gbest_position_d;
    double *gbest_fitness_d;
    int *lock_d;
    int block_size = min(1024, args.blocks_per_grid);

    int dim_d = args.dimensions;
    min_pos = -100.0, max_pos = +100.0;
    w = 1, c1 = 2.0, c2 = 2.0;
    particle_cnt = args.particle_cnt;
    max_v = (max_pos - min_pos) * 1.0;

    HANDLE_ERROR(cudaMalloc((void **)&gbest_position_d, sizeof(double) * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_fitness_d, sizeof(double) * 1));
    gbest = (particle *)malloc(sizeof(particle));
    initialize_gbest(gbest, args.dimensions);
    ParticleInitCoal(p, args.dimensions);
    int dimensions = args.dimensions;
    HANDLE_ERROR(cudaMalloc((void **)&position_d, sizeof(double) * particle_cnt * dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, sizeof(double) * particle_cnt * dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, sizeof(double) * dimensions * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&lock_d, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, sizeof(double) * particle_cnt * dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, sizeof(double) * particle_cnt * dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, sizeof(double) * dimensions * particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_position_d, gbest->position, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(gbest_velocity_d, gbest->velocity, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_fitness_d, gbest->g_fitness, sizeof(double), cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(gbest_d, &gbest, sizeof(particle), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(w_d, &w, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c1_d, &c1, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(c2_d, &c2, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_pos_d, &max_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(min_pos_d, &min_pos, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_v_d, &max_v, sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(max_iter_d, &args.max_iter, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(particle_cnt_d, &args.particle_cnt, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size2, &block_size, sizeof(int)));
    HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));
    clock_t end_init = clock();
    HANDLE_ERROR(cudaEventRecord(start));

    double *best_fitness_buf_d;
    double *best_positions_buf_d;
    HANDLE_ERROR(cudaMalloc((void **)&best_fitness_buf_d,
                            sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&best_positions_buf_d,
                            sizeof(double) * args.blocks_per_grid * dimensions));

    // changes in the kernel for updated fine grained
    int threads_per_block = args.threads_per_block;
    int particle_count = args.particle_cnt;
    dimensions = args.dimensions;
    int num_blocks = args.blocks_per_grid;

    int dimensions_per_thread, particles_per_block; // We are setting these to variables in calculate_launch_params
    calculate_launch_params(particle_count, dimensions,
                            threads_per_block, num_blocks, &dimensions_per_thread,
                            &particles_per_block);

    for (unsigned int i = 0; i < args.max_iter; i++)
    {
        updateParticles<<<num_blocks, threads_per_block, 2 * sizeof(double) * particles_per_block>>>(position_d, velocity_d, fitness_d,
                                                                                                     pbest_pos_d, pbest_fit_d,
                                                                                                     best_fitness_buf_d, best_positions_buf_d,
                                                                                                     dimensions, gbest_position_d, gbest_fitness_d,
                                                                                                     particles_per_block, dimensions_per_thread);

        findGlobalBest<<<1, block_size, 2 * block_size * sizeof(double)>>>(
            best_fitness_buf_d, best_positions_buf_d, particle_cnt, dim_d, gbest_fitness_d, gbest_position_d);
    }
    HANDLE_ERROR(cudaEventRecord(stop));

    HANDLE_ERROR(cudaMemcpy(p->position, position_d, sizeof(double) * particle_cnt * dimensions, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->velocity, velocity_d, sizeof(double) * particle_cnt * dimensions, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->fitness, fitness_d, sizeof(double) * particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->pbest_pos, pbest_pos_d, sizeof(double) * dimensions * particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(p->pbest_fit, pbest_fit_d, sizeof(double) * particle_cnt, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(gbest->g_fitness, gbest_fitness_d, sizeof(double), cudaMemcpyDeviceToHost));

    clock_t end_exe = clock();
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&exe_time, start, stop));
    free(p);
    cudaFree(position_d);
    cudaFree(velocity_d);
    cudaFree(fitness_d);
    cudaFree(pbest_pos_d);
    cudaFree(pbest_fit_d);
    // cudaFree(gbest_d);
    cudaFree(lock_d);

    printf("best result: %lf\n", gbest->g_fitness[0]);
    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    printf("[Cuda Exec time]: %f (sec)\n", exe_time / 1000);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    cudaFree(best_fitness_buf_d);
    cudaFree(best_positions_buf_d);
    free_gbest();

    return 0;
}
