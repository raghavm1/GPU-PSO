#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* getopt */
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
    if (args->threads_per_block && args->particle_cnt && args->dimensions == 1)
        args->blocks_per_grid = ((args->particle_cnt - 1) / args->threads_per_block + 1);
    else if (args->dimensions > 1)
    {
        args->blocks_per_grid = args->particle_cnt;
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
