#include <mpi.h>
#include <math.h>
#include <cstdlib>

#define NUM_P 4
#define RADIUS 1
#define sind(x) (sin(fmod((x),360) * M_PI / 180))
#define cosd(x) (cos(fmod((x),360) * M_PI / 180))

int dboard(int N) {
	int M = 0;
	double len = 1 / sqrt(2);
	for (int i = 0; i < N; i++) {
		// Throw darts at dartboard
		double r = RADIUS * (double)rand() / RAND_MAX;
		double theta = 360 * (double)rand() / RAND_MAX;
		
		// Generate random numbers for X and Y coordinates
		double x = sqrt(r) * cosd(theta);
		double y = sqrt(r) * sind(theta);
		// check whether dart lands in square
		if (x <= len && y <= len && x >= (-1)*len && y >= (-1)*len) {
			M++;
		}
	}
	//printf("M is %d\n", M);
	return M;
}

int main(int argc, char *argv[]) {
	
	// Parse inputs
	int N = atoi(argv[1]);
	int R = atoi(argv[2]);
	//printf("N: %s, R: %s\n", argv[1], argv[2]);
	double t1, t2;
	
	// Set up MPI
	MPI_Init(&argc, &argv);
	
	// Get communicator size and my rank
	MPI_Comm comm = MPI_COMM_WORLD;
	int p, rank, n;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &rank);
	
	/* code */
	// Initial the random number generator
	srand(rank);
	if (rank == 0){
		n = N / p;
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	t1 = MPI_Wtime();
	double sum_pi = 0;
	// Average the M through R rounds.
	for (int i = 0; i < R; i++) {
		double local_m = dboard(n);
		double local_sum = 0;
		MPI_Reduce(&local_m, &local_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

		if (rank == 0) {
			double mpi_pi = 2 * N / local_sum;
			sum_pi += mpi_pi;
			//printf("iteration now is %d\n", i);
			if (i == R-1) {
				double ave_pi = sum_pi / R;
				printf("N = %d, R = %d, P = %d, PI = %f\n", N, R, p, ave_pi);
				t2 = MPI_Wtime();
				printf("Time = %f s\n", t2 - t1);
			}
		}
	}
	
	//finalize MPI..
	MPI_Barrier(comm);
	MPI_Finalize();
	return 0;
}

