#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>  /* getopt */
#include <limits.h>
#include <time.h>

#define MAX_ITERA 100000
#define BlocksPerGrid ((COUNT-1)/ThreadsPerBlock+1)
#define ThreadsPerBlock 1024
#define RND() ((double)rand()/RAND_MAX) // 產生[0,1] 亂數
#define min(x,y) (x)<(y)?(x):(y)
#define COUNT 4096

typedef struct tag_particle_Coal{
    double *position;  // Current position, i.e., x value
    double *velocity;  // Current particle velocity
    double *fitness;   // Fitness function value
    double *pbest_pos; // Particle's current best position
    double *pbest_fit; // Particle's current best fitness value
} particle_Coal;

typedef struct tag_particle{
    double *position;   // Current position, i.e., x value
    double *velocity;   // Current particle velocity
    double *fitness;    // Fitness function value     // defined as a separate device variable
    // double *pbest_pos;  // Particle's current best position
    // double pbest_fit;  // Particle's current best fitness value
} particle;  // this is only for gbest



typedef struct tag_arguments{
    int max_iter;
    int particle_cnt;
    int threads_per_block;
    int blocks_per_grid;
    int dimensions;
    int verbose;
} arguments;

double w, c1, c2;                    // 相關權重參數
double max_v;                        // 最大速度限制
double max_pos, min_pos;             // 最大,小位置限制
unsigned int particle_cnt;               // 粒子數量
// particle gbest = { INT_MIN, INT_MIN, 
//     INT_MIN, INT_MIN, INT_MIN };     // 全域最佳值

// void ParticleInitCoal(particle_Coal *p);
// void ParticleInit(particle *p);

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {        
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);        
        exit(EXIT_FAILURE);    
    }
}

int parseArgs(arguments *args, int argc, char **argv){
    int cmd_opt = 0;

    while(1) {
        // cmd_opt = getopt(argc, argv, "v:m:c:t:b::");
        cmd_opt = getopt(argc, argv, "v:m:c:t:d:");
        if (cmd_opt == -1) {
            break;
        }
        switch (cmd_opt) {
            case 'm':
                args->max_iter = atoi(optarg);
                break;
            case 'c':
                args->particle_cnt = atoi(optarg);
                break;
            case 't':
                args->threads_per_block = atoi(optarg);
                break;
            // case 'b':
            //     args->blocks_per_grid = atoi(optarg);
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

    if(args->threads_per_block && args->particle_cnt)
        args->blocks_per_grid = ((args->particle_cnt-1)/args->threads_per_block+1);
    if(args->verbose){
        printf("max_iter is %d \n", args->max_iter);
        printf("particle_cnt is %d \n", args->particle_cnt);
        printf("threads_per_block is %d \n", args->threads_per_block);
        printf("blocks_per_grid is %d \n", args->blocks_per_grid);
        printf("=============================\n");
    }
    return 1;
}

//------------------------------------------------
// Above is the header file stuff
//------------------------------------------------



#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// cuda prototype
// __host__ __device__ double fit(double x);
__host__ __device__ double fit(double* x, int d);
__global__ void findBestInOneBlock(double *aux, double *aux_pos, int d, double *gbest_position_d, double *gbest_fitness_d);
__global__ void move(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, int *lock, volatile double *aux, volatile double *aux_pos,
    int d, double *gbest_position_d, double *gbest_fitness_d, volatile double* privateBestPosQueue_d, int iter);
void ParticleInitCoal(particle_Coal *p, int d, particle *gbest);
void ParticleFreeCoal(particle_Coal *p);
void ParticleInitGbest(particle *gbest_d, int d);
void ParticleFreeGbest(particle *gbest_d);

//cuda constant memory
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
//cuda function


// HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
// HANDLE_ERROR(cudaMemcpyToSymbol(tile_size2, &block_size, sizeof(int)));

__global__ void findBestInOneBlock(double *aux, double *aux_pos, int d, double *gbest_position_d, double *gbest_fitness_d) {
    int idx = threadIdx.x;  // the block Id is 0.
  // printf("Grid dims are  %d\n", blockDim.x); 
   extern __shared__ double privateBestShit[];
   // __shared__ double privateBest[512]; // assuming not more than 200 blocks are formed in move kernel
   // __shared__ double privateBestIndex[512]; // assuming not more than 200 blocks are formed in move kernel
	    double *privateBest = &privateBestShit[0];
    double *privateBestIndex = &privateBestShit[blockDim.x];
    // tile_size2 is num of blocks in move kernel
    if (idx < tile_size2) {  // this condition should always be true
        privateBest[idx] = aux[idx];
        privateBestIndex[idx] = idx;
    }
    else {
        privateBest[idx] = -1.0;  // Default to an invalid value for unused threads
    }
    __syncthreads();

    // Handle the case where tile_size2 is odd    
    if (tile_size2 % 2!=0 && idx == tile_size2 - 2){ // will not evaluate to true when tile_size2 == 1
        if (privateBest[tile_size2 -2]<privateBest[tile_size2 -1]){
            privateBest[tile_size2 -2] = privateBest[tile_size2 -1];
            privateBestIndex[tile_size2 -2] = tile_size2 -1;
        }
    }
    __syncthreads();

    // Reduction loop
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) { 
        if (idx < stride && idx + stride < tile_size2) {
            if (privateBest[idx] < privateBest[idx + stride]) {
                privateBest[idx] = privateBest[idx + stride];
                privateBestIndex[idx] = idx + stride;
            }
        }
        __syncthreads();
    }

    // Update global best fitness and position
    if (idx == 0) {
        if (privateBest[0] > gbest_fitness_d[0]) {
            gbest_fitness_d[0] = privateBest[0];
            int largest_idx = privateBestIndex[0];
            // printf("gbest_fitness_d[0] %f\n", gbest_fitness_d[0]);
            for (int j = 0; j < d; j++) {
                // gbest_position_d[j] = privateBestPos[j];
                gbest_position_d[j] = aux_pos[largest_idx*d + j];
            }
            __threadfence_block();
        }
    }
}


__global__ void move(double *position_d, double *velocity_d, double *fitness_d, 
    double *pbest_pos_d, double *pbest_fit_d, int *lock, volatile double *aux, volatile double *aux_pos,
    int d, double *gbest_position_d, double *gbest_fitness_d, volatile double* privateBestPosQueue_d, int iter){

    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    extern __shared__ double sharedMemory[];
    double *privateBestQueue = (double *)sharedMemory;
    // double *privateBestPosQueue = (double *)&sharedMemory[tile_size];
    __shared__ unsigned int queue_num;
    if (threadIdx.x == 0) {
        queue_num = 0;
        privateBestQueue[0] = INT_MIN;
    }
    __syncthreads();
    // need to change input params to get info on gbest as separate vectors and variable
    curandState state1, state2;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state1);
    curand_init((unsigned long long)clock() + idx, 0, 0, &state2);
    double fitness = fitness_d[idx];
    if(idx < particle_cnt_d){
        for (int j=0; j< d; j++){
            double v    = velocity_d[idx*d+j];
            double pos  = position_d[idx*d+j];
            double ppos = pbest_pos_d[idx*d+j];
            // if(tidx == 0)
                // queue_num = 0;
            velocity_d[idx*d+j] = w_d * v + c1_d * curand_uniform_double(&state1) * (ppos - pos) 
                + c2_d * curand_uniform_double(&state2) * (gbest_position_d[j] - pos);
            if(velocity_d[idx*d+j] < -max_v_d)
                velocity_d[idx*d+j] = -max_v_d;
            else if(velocity_d[idx*d+j] > max_v_d)
                velocity_d[idx*d+j] = max_v_d;
            position_d[idx*d+j] = pos + v;
            if(position_d[idx*d+j] > max_pos_d)
                position_d[idx*d+j] = max_pos_d; 
            else if(position_d[idx*d+j] < min_pos_d) 
                position_d[idx*d+j] = min_pos_d; 
            }
            fitness = fit(position_d+idx*d, d);
            // printf("fitness %f\n", fitness);
            if(fitness > pbest_fit_d[idx]){
                for(int j=0; j<d; j++){
                    pbest_pos_d[idx*d+j] = position_d[idx*d+j];
                    // printf("I am here 1\n");
                }
                pbest_fit_d[idx] = fitness;
            }
    }
    // privateBestPosQueue[0] = INT_MIN;
    __syncthreads();
    if(fitness > gbest_fitness_d[0]){
        unsigned const my_index = atomicAdd(&queue_num, 1) +1;
        // printf("queue_num: %d\n",queue_num);
        privateBestQueue[my_index] = fitness;
        // if(my_index==0){
        //      printf("my_index: %d\n",my_index);
        // }
        // printf("Thread %d, Fitness: %f, queue_num: %d,  iter: %d\n", idx, fitness, queue_num, iter);
        for(int j=0; j< d; j++){
            privateBestPosQueue_d[blockIdx.x*blockDim.x*d + my_index*d +j] = position_d[idx*d+j];
        }
        // privateBestPosQueue_d[my_index] = pos;
    }
    __syncthreads();
    if(idx < particle_cnt_d){
        if(tidx==0){
            int maxx = -1;
            aux[blockIdx.x] = INT_MIN;
            aux_pos[blockIdx.x] = INT_MIN;
            if(queue_num){
                for(int j=1; j<=queue_num; j++){
                    // if(privateBestQueue[0]<-1000)
                        // printf("privateBestQueue[0] before condition: %f, queue: %d, Thread %d, iter %d\n", privateBestQueue[0], queue_num,idx, iter);
                    if(privateBestQueue[j] >= privateBestQueue[0]){
                    //    privateBestPosQueu[0] = privateBestPosQueue[j];
                        maxx = j;
                        privateBestQueue[0] = privateBestQueue[j];
                        // printf("Thread %d, iter %d, privateBestQueue[0]: %f\n", idx, iter, privateBestQueue[0]);
                    }
                }
                // for(int k=0; k< d; k++){
                //     privateBestPosQueue_d[blockIdx.x*blockDim.x+k] = privateBestPosQueue_d[blockIdx.x*blockDim.x + maxx*d +k];
                // }
                // aux_pos[blockIdx.x] = privateBestPosQueue[0];
                // printf("maxx : %d, Thread %d, iter %d\n", maxx, idx, iter);
                if(maxx!=-1){
                    aux[blockIdx.x] = privateBestQueue[0];
                    // printf("aux[blockIdx.x]: %f\n", aux[blockIdx.x]);
                    // printf("Thread %d, iter %d, here 1\n", idx, iter);
                    for(int k=0; k< d; k++){
                        aux_pos[blockIdx.x*d+k] = privateBestPosQueue_d[blockIdx.x*blockDim.x*d + (maxx)*d +k];
                    }
                }
                // printf("Thread %d, iter %d, here 2\n", idx, iter);
            }
        }
        // position_d[idx] = pos;
        // velocity_d[idx] = v;
        // fitness_d[idx]  = fitness;
    }
}

__host__ __device__ double fit(double* x, int d){
    // x**3 - 0.8x**2 - 1000x + 8000
    // d is num of dimensions
    // x is the array of positions of all particles starting from an offset
    // example we want fitness of 3rd particle when dimension= 20 
    // then expected call is fit(position + 3*20, 20) where position is type *double
    double summation = 0.0;
    for(int i=0; i< d; i++){
        summation += fabs(8000.0 + x[i]*(-10000.0+x[i]*(-0.8+x[i])));
        // summation += 8000.0 + x[i]*(-10000.0+x[i]*(-0.8+x[i]));
    }
    return summation;
}

int main(int argc, char **argv){
    arguments args = {};
    args.max_iter = 10000;
    args.particle_cnt = 4096;
    args.threads_per_block = 128;
    args.dimensions = 64;
    args.blocks_per_grid = ((args.particle_cnt-1)/args.threads_per_block+1);
    args.verbose = 1;
    int retError = parseArgs(&args, argc, argv);
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float exe_time;
    // Variable declarations
    clock_t begin_app  = clock();
    clock_t begin_init = begin_app;
    particle_Coal *p; // p : particle swarm
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

    // double *gbest_best_pos_d;
    // double gbest_best_fit_d;
    
    double *aux, *aux_pos;
    int *lock_d; // block level lock for gbest
    
    // int block_size = min(1024, args.blocks_per_grid); // why is this line here?
    int block_size = args.blocks_per_grid; // block_size i think means number of blocks

    // Set parameters
    min_pos = -100.0 , max_pos = +100.0;  // Position limits, i.e., solution space limits
    w = 1, c1 = 2.0, c2 = 2.0;            // Inertia weight and acceleration constants
    particle_cnt = args.particle_cnt;     // Set the number of particles
    max_v = (max_pos - min_pos) * 1.0;    // Set the maximum velocity limit

    p = (particle_Coal*) malloc(sizeof(particle_Coal));
    gbest = (particle*) malloc(sizeof(particle));
    //p->position  = (double *) malloc(sizeof(double)* particle_cnt);
    //p->velocity  = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->fitness   = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->pbest_pos = (double *) malloc(sizeof(double)* particle_cnt); 
    //p->pbest_fit = (double *) malloc(sizeof(double)* particle_cnt);

    ParticleInitGbest(gbest, args.dimensions); //gbest init must be before particle coal init
    ParticleInitCoal(p, args.dimensions, gbest); // Particle initialization

    printf("Allocating device memory\n");
    //HANDLE_ERROR(cudaMalloc((void **)&p_d, sizeof(particle_Coal)));
    HANDLE_ERROR(cudaMalloc((void **)&position_d, sizeof(double)* particle_cnt* args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&privateBestPosQueue_d, sizeof(double)* particle_cnt* args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&velocity_d, sizeof(double)* particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&fitness_d, sizeof(double)* particle_cnt));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_pos_d, sizeof(double)* particle_cnt * args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&pbest_fit_d, sizeof(double)* particle_cnt));

    // HANDLE_ERROR(cudaMalloc((void **)&gbest_d, sizeof(particle)));// need to change this.. multiple lines
    HANDLE_ERROR(cudaMalloc((void **)&gbest_position_d, sizeof(double)* args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_velocity_d, sizeof(double)* args.dimensions));
    HANDLE_ERROR(cudaMalloc((void **)&gbest_fitness_d, sizeof(double)*3));

    HANDLE_ERROR(cudaMalloc((void**)&lock_d, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&aux, sizeof(double)* args.blocks_per_grid));
    HANDLE_ERROR(cudaMalloc((void **)&aux_pos, sizeof(double)* args.blocks_per_grid * args.dimensions));

    printf("Copying to device\n");
    HANDLE_ERROR(cudaMemcpy(position_d, p->position, sizeof(double)* particle_cnt* args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(velocity_d, p->velocity, sizeof(double)* particle_cnt* args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(fitness_d, p->fitness, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_pos_d, p->pbest_pos, sizeof(double)* particle_cnt* args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pbest_fit_d, p->pbest_fit, sizeof(double)* particle_cnt, cudaMemcpyHostToDevice));

    // HANDLE_ERROR(cudaMemcpy(gbest_d, &gbest, sizeof(particle), cudaMemcpyHostToDevice)); // need to change this.. multiple lines
    HANDLE_ERROR(cudaMemcpy(gbest_position_d, gbest->position, sizeof(double)*args.dimensions, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gbest_velocity_d, gbest->velocity, sizeof(double)*args.dimensions, cudaMemcpyHostToDevice));
    // HANDLE_ERROR(cudaMemcpy(gbest_fitness_d, &gbest->fitness, sizeof(double), cudaMemcpyHostToDevice));
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
    //HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.block_queue_size, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size, &args.threads_per_block, sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol(tile_size2, &block_size, sizeof(int)));
    HANDLE_ERROR(cudaMemset(lock_d, 0, sizeof(int)));
    clock_t end_init = clock();
    clock_t begin_exe  = end_init;
    HANDLE_ERROR(cudaEventRecord(start));
    for(unsigned int i = 0; i < args.max_iter; i++){
        // move<<<args.blocks_per_grid, args.threads_per_block, sizeof(double) * (2 * args.threads_per_block + 1)>>>
        //     (position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d, 
        //     gbest_d, lock_d, aux, aux_pos, args.dimensions, gbest_position_d, gbest_fitness_d);
        move<<<args.blocks_per_grid, args.threads_per_block, sizeof(double) * (args.threads_per_block + 1)>>>(position_d, velocity_d, fitness_d, pbest_pos_d, pbest_fit_d, 
            lock_d, aux, aux_pos, args.dimensions, gbest_position_d, gbest_fitness_d, privateBestPosQueue_d, i);
        //HANDLE_ERROR(cudaEventRecord(stop));
        cudaDeviceSynchronize();
        findBestInOneBlock<<<1, block_size, sizeof(double) * block_size * 2>>>(aux, aux_pos, args.dimensions, gbest_position_d, gbest_fitness_d);
        cudaDeviceSynchronize();
        // printf("iter %d\n", i);                      
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    // HANDLE_ERROR(cudaMemcpy(p->position, position_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(p->velocity, velocity_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(p->fitness, fitness_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(p->pbest_pos, pbest_pos_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(p->pbest_fit, pbest_fit_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(gbest->position, gbest_position_d, sizeof(double)* args.dimensions, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(gbest->velocity, gbest_velocity_d, sizeof(double)* particle_cnt, cudaMemcpyDeviceToHost));
    // HANDLE_ERROR(cudaMemcpy(&gbest, gbest_d, sizeof(particle), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(gbest->fitness, gbest_fitness_d, sizeof(double)*3, cudaMemcpyDeviceToHost));
    clock_t end_exe  = clock();
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&exe_time, start, stop));

    //for(int i=0; i<particle_cnt; i++)
    //    printf("#%d : %lf , %lf . %lf\n", i+1, p->position[i], p->fitness[i], p->velocity[i]);
    // free(p);
    
    // printf("the answer : %10.6lf, %lf\n", -57.469, fit(-57.469));
    // printf("best result: %10.6lf, %lf\n", gbest.position, gbest.fitness);
    printf("best result: %lf\n", gbest->fitness[0]);
    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    //printf("[Execution time]: %lf (sec)\n", (double)(end_exe - begin_exe) / CLOCKS_PER_SEC);
    printf("[Cuda Exec time]: %f (sec)\n", exe_time / 1000);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);
    ParticleFreeCoal(p);
    ParticleFreeGbest(gbest);
    cudaFree(position_d);
    cudaFree(velocity_d);
    cudaFree(fitness_d);
    cudaFree(pbest_pos_d);
    cudaFree(pbest_fit_d);
    // cudaFree(gbest_d);
    cudaFree(lock_d);
    return 0;
}

void ParticleInitCoal(particle_Coal *p, int d, particle *gbest){
    // d is num of dimensions
	unsigned int i;
    unsigned int j;
	const double pos_range = max_pos - min_pos; 
    srand((unsigned)time(NULL));
    p->position  = (double *) malloc(sizeof(double)* particle_cnt * d);
    p->velocity  = (double *) malloc(sizeof(double)* particle_cnt*d); 
    p->fitness   = (double *) malloc(sizeof(double)* particle_cnt); 
    p->pbest_pos = (double *) malloc(sizeof(double)* particle_cnt* d); 
    p->pbest_fit = (double *) malloc(sizeof(double)* particle_cnt);

	for(i=0; i<particle_cnt; i++) {
        for(j=0; j<d; j++) {
		
            p->pbest_pos[i*d+j] = p->position[i*d +j] = RND() * pos_range + min_pos;  // removed
            // p->pbest_pos[i*d+j] = p->position[i*d +j] = 0.0;  // added

            // printf("position value : %f\n", p->position[i*d +j]);
            // printf("RND() in init %f\n", RND()); 
            p->velocity[i*d+j] = RND() * max_v;
        }
        p->pbest_fit[i] = p->fitness[i] = fit(p->position + i*d, d);   // removed
        // p->pbest_fit[i] = p->fitness[i] = 0.0;   // added

        if(i==0 || p->pbest_fit[i] > gbest->fitness[0]){
                for(j=0; j<d; j++) {
                    gbest->position[j] = p->position[i*d+j];      
                    gbest->velocity[j] = p->velocity[i*d+j];            
                    // gbest->fitness[0] = p->pbest_pos[i*d+j];  
                }      
                gbest->fitness[0] = p->fitness[i];   
                // printf("gbest->fitness[0] in init %f\n", gbest->fitness[0]);           
                // gbest->pbest_fit[i] = p->pbest_fit[i];   
            } 
    }
    // gbest->fitness[0] = 0.0;  // added 
}

void ParticleFreeCoal(particle_Coal *p) {
    
    free(p->position);
    free(p->velocity);
    free(p->fitness);
    free(p->pbest_pos);
    free(p->pbest_fit);
    free(p);
}


void ParticleInitGbest(particle *gbest_d, int d) {
    // d is number of dimensions
    // Assuming gbest_d is already allocated
    gbest_d->position = (double *) malloc(sizeof(double) * d);   
    gbest_d->velocity = (double *) malloc(sizeof(double) * d);  
    gbest_d->fitness = (double *) malloc(sizeof(double)*3);  
    // gbest_d->pbest_pos = (double *) malloc(sizeof(double) * d); 
    // no need to initialize above arrays

    gbest_d->fitness[0] = -10000000.0;  // Should not matter, will change in ParticleInitCoal, should be small
    // gbest_d->pbest_fit = -10000000.0;      // Should be small number
}

void ParticleFreeGbest(particle *gbest_d) {

    free(gbest_d->position);
    free(gbest_d->velocity);
    free(gbest_d->fitness);
    free(gbest_d);
}
