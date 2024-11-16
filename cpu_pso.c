#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <unistd.h> /* getopt */

typedef struct tag_arguments
{
    int max_iter;
    int particle_cnt;
    int dimension;
    int threads_per_block;
    int blocks_per_grid;
    int block_queue_size;
    int verbose;
} arguments;

// Modified particle structure to handle N dimensions
typedef struct tag_particle
{
    double *position;  /* Current position vector */
    double *velocity;  /* Current velocity vector */
    double fitness;    /* Fitness value */
    double *pbest_pos; /* Particle's best position vector */
    double pbest_fit;  /* Particle's best fitness value */
} particle;

// Global variables
double w, c1, c2;        /* Weight parameters */
double max_v;            /* Maximum velocity limit */
double max_pos, min_pos; /* Position bounds */
unsigned particle_cnt;   /* Number of particles */
unsigned dimension;      /* Number of dimensions */
particle gbest;          /* Global best particle */

#define RND() ((double)rand() / RAND_MAX)

// N-dimensional fitness function (example: sum of squares)
double fit(double *x, int dim)
{
    double sum = 0;
    for (int i = 0; i < dim; i++)
    {
        // Using the original 1D function for each dimension and summing
        double val = 8000.0 + x[i] * (-10000.0 + x[i] * (-0.8 + x[i]));
        sum += fabs(val);
    }
    return sum;
}

int parseArgs(arguments *args, int argc, char **argv)
{
    int cmd_opt = 0;
    while (1)
    {
        cmd_opt = getopt(argc, argv, "v:m:c:d:t:b:q::");
        if (cmd_opt == -1)
            break;

        switch (cmd_opt)
        {
        case 'm':
            args->max_iter = atoi(optarg);
            break;
        case 'c':
            args->particle_cnt = atoi(optarg);
            break;
        case 'd':
            args->dimension = atoi(optarg);
            break;
        case 't':
            args->threads_per_block = atoi(optarg);
            break;
        case 'b':
            args->blocks_per_grid = atoi(optarg);
            break;
        case 'q':
            args->block_queue_size = atoi(optarg);
            break;
        case 'v':
            args->verbose = atoi(optarg);
            break;
        case '?':
            fprintf(stderr, "Illegal option %c\n", (char)optopt);
            break;
        default:
            fprintf(stderr, "Not supported option\n");
            break;
        }
    }

    if (args->verbose)
    {
        printf("Dimensions: %d\n", args->dimension);
        printf("Max iterations: %d\n", args->max_iter);
        printf("Particle count: %d\n", args->particle_cnt);
        printf("=============================\n");
    }
    return 1;
}

particle *AllocateParticle()
{
    particle *p = (particle *)malloc(sizeof(particle) * particle_cnt);
    for (unsigned i = 0; i < particle_cnt; i++)
    {
        p[i].position = (double *)malloc(sizeof(double) * dimension);
        p[i].velocity = (double *)malloc(sizeof(double) * dimension);
        p[i].pbest_pos = (double *)malloc(sizeof(double) * dimension);
    }

    // Allocate global best arrays
    gbest.position = (double *)malloc(sizeof(double) * dimension);
    gbest.pbest_pos = (double *)malloc(sizeof(double) * dimension);
    gbest.velocity = (double *)malloc(sizeof(double) * dimension);

    return p;
}

void ParticleInit(particle *p)
{
    unsigned i, j;
    const double pos_range = max_pos - min_pos;
    srand((unsigned)time(NULL));

    for (i = 0; i < particle_cnt; i++)
    {
        // Initialize position and velocity vectors
        for (j = 0; j < dimension; j++)
        {
            p[i].position[j] = RND() * pos_range + min_pos;
            p[i].velocity[j] = RND() * max_v;
            p[i].pbest_pos[j] = p[i].position[j];
        }

        // Calculate fitness
        p[i].pbest_fit = p[i].fitness = fit(p[i].position, dimension);

        // Update global best if needed
        if (i == 0 || p[i].pbest_fit > gbest.fitness)
        {
            memcpy(gbest.position, p[i].position, dimension * sizeof(double));
            memcpy(gbest.pbest_pos, p[i].pbest_pos, dimension * sizeof(double));
            gbest.fitness = p[i].fitness;
            gbest.pbest_fit = p[i].pbest_fit;
        }
    }
    printf("Best init value comes out to be %lf \n", gbest.fitness);
}

int ParticleMove(particle *p)
{
    unsigned i, j;
    int flags = 0;

    for (i = 0; i < particle_cnt; i++)
    {
        // Update velocity and position for each dimension
        for (j = 0; j < dimension; j++)
        {
            // Update velocity
            p[i].velocity[j] = w * p[i].velocity[j] +
                               c1 * RND() * (p[i].pbest_pos[j] - p[i].position[j]) +
                               c2 * RND() * (gbest.position[j] - p[i].position[j]);

            // Clamp velocity
            if (p[i].velocity[j] > max_v)
                p[i].velocity[j] = max_v;
            if (p[i].velocity[j] < -max_v)
                p[i].velocity[j] = -max_v;

            // Update position
            p[i].position[j] += p[i].velocity[j];

            // Clamp position
            if (p[i].position[j] > max_pos)
                p[i].position[j] = max_pos;
            if (p[i].position[j] < min_pos)
                p[i].position[j] = min_pos;
        }

        // Update fitness
        p[i].fitness = fit(p[i].position, dimension);

        // Update particle's best
        if (p[i].fitness > p[i].pbest_fit)
        {
            p[i].pbest_fit = p[i].fitness;
            memcpy(p[i].pbest_pos, p[i].position, dimension * sizeof(double));
        }

        // Update global best
        if (p[i].fitness > gbest.fitness)
        {
            memcpy(gbest.position, p[i].position, dimension * sizeof(double));
            gbest.fitness = p[i].fitness;
            flags += 1;
        }
    }
    return flags;
}

void ParticleRelease(particle *p)
{
    for (unsigned i = 0; i < particle_cnt; i++)
    {
        free(p[i].position);
        free(p[i].velocity);
        free(p[i].pbest_pos);
    }
    free(gbest.position);
    free(gbest.pbest_pos);
    free(gbest.velocity);
    free(p);
}

void ParticleDisplay(particle *p)
{
    printf("Best solution found:\n");
    printf("Fitness: %lf\n", gbest.fitness);
    printf("Position: ");
    for (unsigned i = 0; i < dimension; i++)
    {
        printf("%lf ", gbest.position[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    arguments args = {10000, 1024, 4, 128, 4, 10, 1}; // Default dimension is 3
    parseArgs(&args, argc, argv);

    clock_t begin_app = clock();
    clock_t begin_init = begin_app;

    // Initialize parameters
    dimension = args.dimension;
    min_pos = -100.0;
    max_pos = +100.0;
    w = 1;
    c1 = 2.0;
    c2 = 2.0;
    particle_cnt = args.particle_cnt;
    max_v = (max_pos - min_pos) * 1.0;

    // Run PSO
    particle *p = AllocateParticle();
    ParticleInit(p);

    clock_t end_init = clock();
    clock_t begin_exe = end_init;

    for (int i = 0; i < args.max_iter; i++)
    {
        ParticleMove(p);
    }

    clock_t end_exe = clock();

    ParticleDisplay(p);
    ParticleRelease(p);

    printf("[Initial   time]: %lf (sec)\n", (double)(end_init - begin_init) / CLOCKS_PER_SEC);
    printf("[Execution time]: %lf (sec)\n", (double)(end_exe - begin_exe) / CLOCKS_PER_SEC);
    printf("[Elapsed   time]: %lf (sec)\n", (double)(clock() - begin_app) / CLOCKS_PER_SEC);

    return 0;
}