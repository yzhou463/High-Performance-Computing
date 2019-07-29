# Instruction
This project aims to develop a parallel back-tracking algorithm to the classic n-queens problem. Consider a n X n chessboard on which n queens are to be placed. Let (i, j) and (k, l) denote the respective
positions of two queens. The queens are said to threaten each other if.
i = k, or.
j = l, or.
|i - k| = |j - l|.
The n-queens problem is to position the n queens on the chessboard such that no two queens threaten each other.

## Code structure

All the code is located at the root level of the project.

There are multiple header and .cpp files, your implementation will go
into the following file:

- `solver.cpp`: Sequential algorithm for nqueen problem, the master and worker functions for parallel nqueen according
  to the function declarations in `solver.h`


Other files containing code that you should not change are:
- `solver.h`: Declares the nqueen functions.
- `utils.h` and `utils.cpp`: Implements common utility functions.
- `main.cpp`: Implements code for the main executable `nqueen`. This does
  input/output reading and calling of the actual functions.


## Compiling

In order to compile everything, simply run
```sh
make
```


## Running
For running on local system, do:
```sh
mpirun -np <num_procs> ./nqueen <n> <k>
```


For running on the PACE cluster, do:
```sh
qsub -v p=<num_procs>,n=<number_of_queens>,k=<k> pbs_script.pbs
```
