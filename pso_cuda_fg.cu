#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>
#include "common.h"
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

__host__ __device__ double fit(double x)
{
    return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x)));
}

__global__ void updateParticles(double *position_d, double *velocity_d, double *fitness_d,
                                double *pbest_pos_d, double *pbest_fit_d,
                                double *best_fitness_buf, double *best_positions_buf, int dim_d, double *gbest_position_d, double *gbest_fitness_d)
{
    int particle_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    extern __shared__ double shared_mem[];
    double *particle_pos = &shared_mem[0];
    double *particle_vel = &shared_mem[dim_d];
    double *particle_pbest = &shared_mem[2 * dim_d];
    int idx = particle_idx * dim_d + dim_idx;

    if (dim_idx < dim_d)
    {

        particle_pos[dim_idx] = position_d[idx];
        particle_vel[dim_idx] = velocity_d[idx];
        particle_pbest[dim_idx] = pbest_pos_d[idx];
    }
    __syncthreads();

    if (dim_idx < dim_d)
    {

        curandState state;
        curand_init((unsigned long long)clock() + idx, 0, 0, &state);

        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);
        double v = particle_vel[dim_idx];

        double pos = particle_pos[dim_idx];

        double pbest = double(particle_pbest[dim_idx]);

        double gbest_pos = double(gbest_position_d[dim_idx]);

        v = w_d * v +
            c1_d * r1 * (pbest - pos) +
            c2_d * r2 * (gbest_pos - pos);

        v = max(min(v, max_v_d), -max_v_d);

        pos = pos + v;

        pos = max(min(pos, max_pos_d), min_pos_d);

        particle_pos[dim_idx] = pos;
        particle_vel[dim_idx] = v;

        position_d[idx] = pos;
        velocity_d[idx] = v;
    }
    __syncthreads();

    if (threadIdx.x == 0 && particle_idx < particle_cnt_d)
    {
        double fitness = 0.0;

        for (int i = 0; i < dim_d; i++)
        {
            fitness += fit(particle_pos[i]);
        }

        fitness_d[particle_idx] = fitness;

        if (fitness > pbest_fit_d[particle_idx])
        {
            pbest_fit_d[particle_idx] = fitness;

            // Store all dimensions of this particle's position
            for (int i = 0; i < dim_d; i++)
            {
                pbest_pos_d[particle_idx * dim_d + i] = particle_pos[i];
            }
        }

        // Store fitness and all position dimensions for global best finding
        best_fitness_buf[particle_idx] = fitness;
        for (int i = 0; i < dim_d; i++)
        {
            best_positions_buf[particle_idx * dim_d + i] = particle_pos[i];
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
    arguments args = {10000, 1024, 1024, 4, 3, 4};
    int retError = pargeArgs(&args, argc, argv);
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float exe_time;
    clock_t begin_app = clock();
    clock_t begin_init = begin_app;
    particle_Coal *p; // p : 粒子群
    p = (particle_Coal *)malloc(sizeof(particle_Coal));
    double *position_d;
    double *velocity_d;
    double *fitness_d;
    double *pbest_pos_d;
    double *pbest_fit_d;
    double *aux, *aux_pos;
    double *gbest_position_d;
    double *gbest_fitness_d;
    int *lock_d; // block level lock for gbest
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
    HANDLE_ERROR(cudaMemcpy(gbest_fitness_d, gbest->g_fitness, sizeof(double), cudaMemcpyHostToDevice));
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
                            sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&best_positions_buf_d,
                            sizeof(double) * particle_cnt * dimensions));

    size_t particle_shared_mem = 3 * dimensions * sizeof(double);

    size_t reduction_shared_mem = 2 * block_size * sizeof(double);

    for (unsigned int i = 0; i < args.max_iter; i++)
    {

        updateParticles<<<particle_cnt, dimensions, particle_shared_mem>>>(
            position_d, velocity_d, fitness_d,
            pbest_pos_d, pbest_fit_d,
            best_fitness_buf_d, best_positions_buf_d, dim_d, gbest_position_d, gbest_fitness_d);

        // Find global best
        findGlobalBest<<<1, block_size, reduction_shared_mem>>>(
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
