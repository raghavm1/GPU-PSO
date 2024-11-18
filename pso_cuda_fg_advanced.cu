#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* getopt */
#include <limits.h>
#include <time.h>
#include <float.h>

#define MAX_ITERA 100000
#define BlocksPerGrid ((COUNT - 1) / ThreadsPerBlock + 1)
#define ThreadsPerBlock 1024
#define RND() ((double)rand() / RAND_MAX) // 產生[0,1] 亂數
#define min(x, y) (x) < (y) ? (x) : (y)
#define COUNT 4096

// typedef struct tag_particle_Coal{
//     double position[COUNT];  // 目前位置, 即x value
//     double velocity[COUNT];  // 目前粒子速度
//     double fitness[COUNT] ;  // 適應函式值
//     double pbest_pos[COUNT]; // particle 目前最好位置
//     double pbest_fit[COUNT]; // particle 目前最佳適應值
// } particle_Coal;

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
    // Allocate memory for position and pbest_pos
    gbest->position = (double *)malloc(sizeof(double) * dimensions);
    gbest->velocity = NULL; // If velocity is not needed, set it to NULL
    gbest->pbest_pos = (double *)malloc(sizeof(double) * dimensions);

    // Initialize fitness and pbest_fit
    gbest->g_fitness = (double *) malloc(sizeof(double)*1);  
    gbest->g_fitness[0] = -DBL_MAX;
    gbest->pbest_fit = -DBL_MAX;

    // Optionally initialize position and pbest_pos to some default values
    for (int i = 0; i < dimensions; i++)
    {
        gbest->position[i] = 0.0;  // Default to 0.0
        gbest->pbest_pos[i] = 0.0; // Default to 0.0
    }
}

void free_gbest()
{
    free(gbest->position);
    free(gbest->pbest_pos);
}

// void ParticleInitCoal(particle_Coal *p);
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
    // fprintf(stderr, "argc:%d\n", argc);
    while (1)
    {
        // fprintf(stderr, "proces index:%d\n", optind);
        cmd_opt = getopt(argc, argv, "v:m:c:t:b:d:");
        /* End condition always first */
        if (cmd_opt == -1)
        {
            break;
        }
        /* Print option when it is valid */
        // if (cmd_opt != '?') {
        //     fprintf(stderr, "option:-%c\n", cmd_opt);
        // }
        /* Lets parse */
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
            if (optarg == NULL) {
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
    // Do we have args?
    // if (argc > optind) {
    //    int i = 0;
    //    for (i = optind; i < argc; i++) {
    //        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    //    }
    //}

    // TODO: check there, when threads_per_block is bigger than particle_cnt
    // the block level atomic lock will be fail
    // [Result]: save value to shared memory should check that value is valid
    //           or just set threads_per_block is bigger than particle_cnt
    // if(args->threads_per_block > args->particle_cnt)
    //    args->threads_per_block = args->particle_cnt;
    // if (args->threads_per_block && args->particle_cnt && args->dimensions == 1)
    //     args->blocks_per_grid = ((args->particle_cnt - 1) / args->threads_per_block + 1);
    // else if (args->dimensions > 1)
    // {
    //     args->blocks_per_grid = args->particle_cnt;
    // }
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

// Additional constant for dimensions
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


// Helper function to calculate launch parameters
void calculate_launch_params(int particle_count, int dimensions, 
                           int threads_per_block, int num_blocks, int* dimensions_per_thread, 
                           int* particles_per_block) {
    
    
    *particles_per_block = (particle_count + num_blocks -1) / num_blocks; // this way does ceil(particle_count/num_blocks)

    int num_dimensions_per_block = *particles_per_block * dimensions;

    *dimensions_per_thread = (num_dimensions_per_block + threads_per_block - 1) / threads_per_block; // similarly this also is ceil
    // int elements_per_thread = (total_elements + total_threads - 1) / total_threads; 
}

// CUDA kernels

__host__ __device__ double fit(double x)
{
    // x**3 - 0.8x**2 - 1000x + 8000
    return fabs(8000.0 + x * (-10000.0 + x * (-0.8 + x)));
    // return 8000.0 + x * (-10000.0 + x * (-0.8 + x));
}

__global__ void updateParticles(double *position_d, double *velocity_d, double *fitness_d,
                              double *pbest_pos_d, double *pbest_fit_d,
                              double *best_fitness_buf_d, double *best_positions_buf_d, 
                              int dim_d, double *gbest_position_d, double *gbest_fitness_d,
                              int particles_per_block, int dimensions_per_thread)
{
    // extern __shared__ double shared_mem[];
    
    __shared__ double privateBestQueue[512]; // assuming not more than 1024 particles per block
    __shared__ double privateBestQueueIndex[512]; // assuming not more than 1024 particles per block

    // double *privateBestQueue = (double *)sharedMemory;
    // double *privateBestPosQueue = (double *)&sharedMemory[tile_size];
    __shared__ unsigned int queue_num;
    if (threadIdx.x == 0) {
        queue_num = 0;
        privateBestQueue[0] = INT_MIN;
    }
    __syncthreads();

    // Calculate which particles this block handles
    int first_particle = blockIdx.x * particles_per_block; // id of first particle in block
    int thread_idx = threadIdx.x; // id of thread wrt to corresponding block
    int total_threads = blockDim.x; // number of threads in block
    
    // Shared memory organization
    // Each particle needs space for position, velocity, and pbest
    int particle_stride = dim_d;  // stride between particles in shared memory
    // double *shared_positions = &shared_mem[0];
    // double *shared_velocities = &shared_mem[particles_per_block * particle_stride];
    // double *shared_pbests = &shared_mem[2 * particles_per_block * particle_stride];
    
    // Calculate work distribution
    int total_elements = particles_per_block * dim_d;
    // int elements_per_thread = (total_elements + total_threads - 1) / total_threads;
    int elements_per_thread = dimensions_per_thread ;
    int start_element = thread_idx * elements_per_thread;  // start_element wrt to block
    int end_element = min(start_element + elements_per_thread, total_elements); // end_element wrt to block
    
    // Process assigned elements
    for (int i = start_element; i < end_element; i++) {
        int particle_offset = i / dim_d;  // Which particle in this block
        int dim_offset = i % dim_d;       // Which dimension of the particle
        int global_particle_idx = first_particle + particle_offset;
        
        if (global_particle_idx < particle_cnt_d) {
            int global_idx = global_particle_idx * dim_d + dim_offset;
            // int shared_idx = particle_offset * particle_stride + dim_offset;
            
            curandState state;
            curand_init((unsigned long long)clock() + global_idx, 0, 0, &state);
            
            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            
            // Update velocity
            double v = w_d * velocity_d[global_idx] +
                      c1_d * r1 * (pbest_pos_d[global_idx] - position_d[global_idx]) +
                      c2_d * r2 * (gbest_position_d[dim_offset] - position_d[global_idx]);
            
            // Clamp velocity
            v = max(min(v, max_v_d), -max_v_d);
            
            // Update position
            double pos = position_d[global_idx] + v;
            pos = max(min(pos, max_pos_d), min_pos_d);
            
            // // Store in shared memory
            // shared_positions[shared_idx] = pos;
            // shared_velocities[shared_idx] = v;
            
            // Store in global memory
            position_d[global_idx] = pos;
            velocity_d[global_idx] = v;
        }
    }

    __syncthreads();

    // Compute fitness for each particle (one thread per particle in block)
    double fitness = 0.0;
    if((thread_idx+1)*elements_per_thread % dim_d ==0){
        start_element = thread_idx * elements_per_thread;
        int new_end_element = min(start_element + dim_d, total_elements); // this new_end_element is end elemt of entire particle

        for (int i = start_element; i < end_element; i++) {
            int particle_offset = i / dim_d;  // Which particle in this block
            int dim_offset = i % dim_d;       // Which dimension of the particle
            int global_particle_idx = first_particle + particle_offset;
            int global_idx = global_particle_idx * dim_d + dim_offset;
            fitness += fit(position_d[global_idx]);
        }

        int particle_offset = start_element / dim_d;
        int global_particle_idx = first_particle + particle_offset;
        if(fitness > pbest_fit_d[global_particle_idx]){
            pbest_fit_d[global_particle_idx] = fitness;

            for (int i = start_element; i < new_end_element; i++) {
                // particle_offset = i / dim_d;  
                int dim_offset = i % dim_d;       
                // global_particle_idx = first_particle + particle_offset;
                int global_idx = global_particle_idx * dim_d + dim_offset;
                pbest_pos_d[global_idx] = position_d[global_idx];
            }
        }

        if(fitness > gbest_fitness_d[0]){
            // printf("fitness : %f\n", fitness);
            unsigned const my_index = atomicAdd(&queue_num, 1) +1;
            privateBestQueue[my_index] = fitness;
            privateBestQueueIndex[my_index] = global_particle_idx;
        }
    }

    __syncthreads();

    if(thread_idx==0){
        int maxx = -1; // keeps track of max index
        best_fitness_buf_d[blockIdx.x] = INT_MIN;
        best_positions_buf_d[blockIdx.x] = INT_MIN; // this is useless actually
        if(queue_num){
            for(int j=1; j<=queue_num; j++){
                if(privateBestQueue[j] >= privateBestQueue[0]){
                    maxx = j;
                    privateBestQueue[0] = privateBestQueue[j];
                }
            }
            if(maxx!=-1){
                best_fitness_buf_d[blockIdx.x] = privateBestQueue[0];
                for(int k=0; k< dim_d; k++){
                    int global_particle_idx = privateBestQueueIndex[maxx];
                    best_positions_buf_d[blockIdx.x*dim_d +k] = position_d[global_particle_idx + k];
                    // privateBestQueueIndex[maxx] is the global_particle_idx of the 'maxx' index
                }
            }
        }

    }

}

__global__ void findGlobalBest(double *best_fitness_buf, double *best_positions_buf, int particle_count, int dim_d, double *gbest_fitness_d, double *gbest_position_d)
{
    extern __shared__ double shared_data[];
    double *s_fitness = &shared_data[0];          // blockDim.x elements
    double *s_indices = &shared_data[blockDim.x]; // blockDim.x elements

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load fitness values and indices into shared memory
    if (gid < particle_count)
    {
        s_fitness[tid] = best_fitness_buf[gid];
        s_indices[tid] = gid; // Store particle index
    }
    else
    {
        s_fitness[tid] = -INFINITY;
        s_indices[tid] = -1;
    }
    __syncthreads();

    // Reduction to find maximum fitness while keeping track of particle index
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

    // Thread 0 updates global best if necessary
    if (tid == 0)
    {
        if (s_fitness[0] > gbest_fitness_d[0])
        {
            //printf("found something better");
            int best_idx = (int)s_indices[0];
            if (best_idx >= 0)
            {
                gbest_fitness_d[0] = s_fitness[0];
                // printf("gbest_fitness_d[0] : %f\n", gbest_fitness_d[0]);

                // Copy all dimensions of the best position
                for (int d = 0; d < dim_d; d++)
                {
                    gbest_position_d[d] = best_positions_buf[best_idx * dim_d + d];
                }
            }
        }
    }
}

// Modified initialization function
void ParticleInitCoal(particle_Coal *p, int dimensions)
{
    const double pos_range = max_pos - min_pos;
    srand((unsigned)time(NULL));

    // Allocate memory for N-dimensional particles
    p->position = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->velocity = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->fitness = (double *)malloc(sizeof(double) * particle_cnt);
    p->pbest_pos = (double *)malloc(sizeof(double) * particle_cnt * dimensions);
    p->pbest_fit = (double *)malloc(sizeof(double) * particle_cnt);

    for (int i = 0; i < particle_cnt; i++)
    {
        double fitness = 0.0;

        // Initialize each dimension
        for (int d = 0; d < dimensions; d++)
        {
            int idx = i * dimensions + d;
            p->position[idx] = double(RND() * pos_range + min_pos);
            // p->position[idx] = 0.0;
            p->velocity[idx] = RND() * max_v;
            p->pbest_pos[idx] = double(p->position[idx]);

            // Accumulate fitness across dimensions
            fitness += fit(p->position[idx]);
        }

        p->fitness[i] = fitness;
        p->pbest_fit[i] = fitness;
        // Update global best if necessary
        if (fitness > gbest->g_fitness[0])
        {
            gbest->g_fitness[0] = fitness;
            for (int d = 0; d < dimensions; d++)
            {
                gbest->position[d] = p->position[i * dimensions + d];
            }
        }
    }
    //gbest->g_fitness[0] = 0;
    // printf("GBest fitness value:  %lf\n", gbest->g_fitness[0]);
}

int main(int argc, char **argv)
{
    // arguments args = {10000, 1024, 1024, 4, 3, 4};
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
    double *gbest_velocity_d;
    double *gbest_fitness_d;
    int *lock_d; // block level lock for gbest
    int block_size = min(1024, args.blocks_per_grid);

    int dim_d = args.dimensions;
    // 設定參數
    min_pos = -100.0, max_pos = +100.0; // 位置限制, 即解空間限制
    w = 1, c1 = 2.0, c2 = 2.0;          // 慣性權重與加速常數設定
    particle_cnt = args.particle_cnt;   // 設粒子個數
    max_v = (max_pos - min_pos) * 1.0;  // 設最大速限

    // p = (particle_Coal *)malloc(sizeof(particle_Coal));
    //  p->position  = (double *) malloc(sizeof(double)* particle_cnt);
    //  p->velocity  = (double *) malloc(sizeof(double)* particle_cnt);
    //  p->fitness   = (double *) malloc(sizeof(double)* particle_cnt);
    //  p->pbest_pos = (double *) malloc(sizeof(double)* particle_cnt);
    //  p->pbest_fit = (double *) malloc(sizeof(double)* particle_cnt);
    //initialize_gbest(gbest, args.dimensions);
    HANDLE_ERROR(cudaMalloc((void **)&gbest_position_d, sizeof(double) * args.dimensions));
    //HANDLE_ERROR(cudaMalloc((void **)&gbest_velocity_d, sizeof(double) * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_fitness_d, sizeof(double) * 1));
    gbest = (particle*) malloc(sizeof(particle));
    initialize_gbest(gbest, args.dimensions);
    ParticleInitCoal(p, args.dimensions); // 粒子初始化
    int dimensions = args.dimensions;
    printf("Allocating device memory\n");
    particle *gbest_d;
    // HANDLE_ERROR(cudaMalloc((void **)&p_d, sizeof(particle_Coal)));
    HANDLE_ERROR(cudaMalloc((void **)&position_d, sizeof(double) * particle_cnt * dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, sizeof(double) * particle_cnt * dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, sizeof(double) * dimensions * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(double) * particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&lock_d, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(double) * args.blocks_per_grid));
    printf("[Original] Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, sizeof(double) * particle_cnt * dimensions, cudaMemcpyHostToDevice));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, sizeof(double) * particle_cnt * dimensions, cudaMemcpyHostToDevice));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, sizeof(double) * dimensions * particle_cnt, cudaMemcpyHostToDevice));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(double) * particle_cnt, cudaMemcpyHostToDevice));
    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(gbest_position_d, gbest->position, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(gbest_velocity_d, gbest->velocity, sizeof(double) * args.dimensions, cudaMemcpyHostToDevice));
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
    // HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.block_queue_size, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size2, &block_size, sizeof(int)));
    HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));
    printf("Copied all !\n");
    clock_t end_init = clock();
    clock_t begin_exe = end_init;
    HANDLE_ERROR(cudaEventRecord(start));

    // Buffers for finding global best
    double *best_fitness_buf_d;
    double *best_positions_buf_d;
    HANDLE_ERROR(cudaMalloc((void **)&best_fitness_buf_d,
                            sizeof(double) * args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&best_positions_buf_d,
                            sizeof(double) * args.blocks_per_grid * dimensions));

    // Shared memory size for updateParticles kernel
    // size_t particle_shared_mem = 3 * dimensions * sizeof(double);
    // size_t particle_shared_mem = 3 * dimensions * sizeof(double);

    // Shared memory size for findGlobalBest kernel (fitness values + particle indices)
    // size_t reduction_shared_mem = 2 * block_size * sizeof(double);


    // changes in the kernel for updated fine grained
    int threads_per_block = args.threads_per_block;
    int particle_count = args.particle_cnt;
    dimensions = args.dimensions;
    int num_blocks = args.blocks_per_grid;

    int dimensions_per_thread, particles_per_block; // We are setting these to variables in calculate_launch_params
    calculate_launch_params(particle_count, dimensions, 
                            threads_per_block, num_blocks, &dimensions_per_thread, 
                            &particles_per_block);
    // size_t shared_mem_size = 3 * particles_per_block * dimensions * sizeof(double);
                
    for (unsigned int i = 0; i < args.max_iter; i++)
    {
        // Update all particles
        // printf("Still inside host, getting into the kernels now...;");
        // updateParticles<<<particle_cnt, dimensions, 3 * dimensions * sizeof(double);>>>(
        //     position_d, velocity_d, fitness_d,
        //     pbest_pos_d, pbest_fit_d,
        //     best_fitness_buf_d, best_positions_buf_d, dim_d, gbest_position_d, gbest_fitness_d);

        updateParticles<<<num_blocks, threads_per_block>>>(position_d, velocity_d, fitness_d,
                              pbest_pos_d, pbest_fit_d,
                              best_fitness_buf_d, best_positions_buf_d, 
                              dimensions, gbest_position_d, gbest_fitness_d,
                              particles_per_block, dimensions_per_thread);

        // Find global best
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

    // for(int i=0; i<particle_cnt; i++)
    //     printf("#%d : %lf , %lf . %lf\n", i+1, p->position[i], p->fitness[i], p->velocity[i]);
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
    // printf("[Execution time]: %lf (sec)\n", (double)(end_exe - begin_exe) / CLOCKS_PER_SEC);
    printf("[Cuda Exec time]: %f (sec)\n", exe_time / 1000);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    cudaFree(best_fitness_buf_d);
    cudaFree(best_positions_buf_d);
    free_gbest();

    return 0;
}