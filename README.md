# PSO algorithm on GPU

## Running the code

First, load cuda-12.4 using the command `module load cuda-12.4`

There are 3 files, one requires a common.h header file.
Also, all these files have default arguments set, so you don't need to pass any arguments.

But if you'd like, I've added flag info and sample arguments to work with.

Arguments 

-m (max iterations)
-c (particle count)
-d (dimensions)
-t (threads per block)
-b (blocks)
-v (verbose, to output argument values when running the code)

### Coarse grained

Compile this file via `nvcc -o coarse pso_cuda_coarse.cu`
Sample command - `./coarse -m 10000 -d 4 -t 64 -c 4096 -v 1`

### Fine-grained

Compile this file via `nvcc -o finegrained pso_cuda_fg.cu`
Sample command - `./finegrained -m 10000 -d 4`

### Advanced fine-grained

Compile this file via `nvcc -o adv pso_cuda_fg_advanced.cu`
Sample command - `./adv -m 10000 -c 4096 -d 4 -t 4 -b 2048 -v 1`

