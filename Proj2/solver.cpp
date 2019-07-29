#include "solver.h"


/*************************** DECLARE YOUR HELPER FUNCTIONS HERE ************************/
/* Check if putting queen on (row, col) is valid. */
bool isValid(const std::vector<unsigned int> &queen_col_pos, 
			 const unsigned int &row, 
			 const unsigned int &col) {
	for (unsigned int i = 0; i < row; i++) {
		if (queen_col_pos[i] == col || abs(queen_col_pos[i] - col) == (row - i)) {
			return false;
		}
	}
	return true;
}

/* The actual sequential solver */
void actual_seq_solver(std::vector<std::vector<unsigned int>> &all_solns, 
					   std::vector<unsigned int> &curr_res, 
					   unsigned int row, 
					   const unsigned int &n) {
	/* Base case */
	if (row == n) {
		all_solns.push_back(curr_res);
		return;
	}

	for (unsigned int col = 0; col < n; col++) {
		if (isValid(curr_res, row, col)) {
			curr_res[row] = col;
			actual_seq_solver(all_solns, curr_res, row + 1, n);
			curr_res[row] = 0;
		}
	}
}

/* Calculate the next partial solution with k queens */
bool next_partial_solution(std::vector<unsigned int> &partial_solution, 
						   std::vector<unsigned int> &curr_res, 
						   unsigned int row,
						   const unsigned int &n, 
						   const unsigned int &k,
						   bool &had_sol) {

	/* Handle edge case of k = 0 */
	if (k == 0 && had_sol) {
		return false;
	} else if (k == 0) {
		partial_solution = curr_res;
		had_sol = true;
		return true;
	}

	/* Base case */
	if (row == k) {
		partial_solution = curr_res;
		had_sol = true;
		return true;
	}

	/* Try each position in current row to find all partial solutions */
	for (unsigned int col = (row == k - 1 && had_sol ? curr_res[row] + 1 : curr_res[row]); col < n; col++) {
		if (isValid(curr_res, row, col)) {
			curr_res[row] = col;

			if (next_partial_solution(partial_solution, curr_res, row + 1, n, k, had_sol)) {
				return true;
			}
		}
	}

	/* If the for loop has finished, then clear the current row and the had_sol flags, then return false. */
	curr_res[row] = 0;
	had_sol = false;
	return false;
}

/* Check if all workers are idle */
bool check_workers_idle(std::vector<bool> &flags, const int &n) {
	return std::all_of(flags.begin(), flags.end(), [](bool working) { return working == false; });
}

/* The solver for the workers */
void worker_solver(std::vector<unsigned int> &curr_res, 
				   unsigned int row, 
				   const unsigned int &n) {
	/* Base case: Full solution found, send it to Master */
	if (row == n) {
		MPI_Send(&curr_res[0], n, MPI_INT, 0, 101, MPI_COMM_WORLD);
		return;
	}

	for (unsigned int col = 0; col < n; col++) {
		if (isValid(curr_res, row, col)) {
			curr_res[row] = col;
			worker_solver(curr_res, row + 1, n);
			curr_res[row] = 0;
		}
	}
}

/*************************** solver.h functions ************************/
void seq_solver(unsigned int n, std::vector<std::vector<unsigned int>> &all_solns) {
	/* Create current result */
	std::vector<unsigned int> curr_res (n, 0);

	/* Start actual solver */
	actual_seq_solver(all_solns, curr_res, 0, n);
}


void nqueen_master(unsigned int n,
				   unsigned int k,
				   std::vector<std::vector<unsigned int> >& all_solns) {

	/* Handle edge case of n = k */
	if (n <= k) {
		seq_solver(n, all_solns);
		return;
	}

	int num_procs; // number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Status stat;
	MPI_Request req;
	int worker;
	int num_workers = num_procs - 1; // number of workers

	/* Create current result */
	std::vector<unsigned int> curr_res (n, 0);

	/* Next partial solution */
	std::vector<unsigned int> partial_solution (n, 0);

	/* Sending buffer*/
	std::vector<unsigned int> res_send (n, 0);
	/* Receiving buffer */
	std::vector<unsigned int> res_recv (n, 0);

	/* Flags */
	bool had_sol = false, found_sol = false, found_all_partial_sol = false, all_work_done = false;
	

	/* Working flags for workers, false -> idle, true -> working */
	std::vector<bool> working (num_workers, false);

	/******************* STEP 1: Send one partial solution to each worker ********************/
	/* Find next partial solution for each worker, send them to the workers */
	for (int w = 1; w < num_procs; w++) {
		/* Find next partial solution */
		if (next_partial_solution(partial_solution, curr_res, 0, n, k, had_sol)) {
			MPI_Send(&partial_solution[0], n, MPI_INT, w, 100, MPI_COMM_WORLD);
			working[w-1] = true;
		} else {
			found_all_partial_sol = true; /* All the partial solutions have been found */
			break;
		}
	}

	/******************* STEP 2: Send partial solutions to workers as they respond ********************/
	while (!found_all_partial_sol || !all_work_done) {
		/* Non-blocking receive data from workers */
		MPI_Irecv(&res_recv[0], n, MPI_INT, MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &req);
		
		/* Find next partial solution */
		if (!found_all_partial_sol) {
			found_sol = next_partial_solution(partial_solution, curr_res, 0, n, k, had_sol);
			found_all_partial_sol = !found_sol;
		}

		/* Wait for next available worker */
		MPI_Wait(&req, &stat);
		worker = stat.MPI_SOURCE;

		/* The worker has found a solution, need the while loop to avoid skipping any partial solution */
		while (res_recv[0] != n) {
			/* Store the result */
			all_solns.push_back(res_recv);

			/* Wait for the next message from worker */
			MPI_Recv(&res_recv[0], n, MPI_INT, MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &stat);
			worker = stat.MPI_SOURCE;
		}

		/* Otherwise, The worker has done the job */
		working[worker-1] = false;
		/* Send a new partial solution if exist. If all partial solutions are found, check if all workers are done. */
		if (!found_all_partial_sol) {
			MPI_Send(&partial_solution[0], n, MPI_INT, worker, 100, MPI_COMM_WORLD);
			working[worker-1] = true;
		} else {
			all_work_done = check_workers_idle(working, num_workers);
		}
	}

	/********************** STEP 3: Terminate **************************/
	/* Send termination message to all workers */
	for (int w = 1; w < num_procs; w++) {
		res_send[k] = 100;
		MPI_Send(&res_send[0], n, MPI_INT, w, 100, MPI_COMM_WORLD);
	}
}

void nqueen_worker(	unsigned int n,
					unsigned int k) {

	/* Handle edge case of n = k */
	if (n <= k) {
		return;
	}

	int proc_id; /* number of processors */
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
	
	/* Receiving buffer */
	std::vector<unsigned int> res_recv (n, 0);

	while (true) {
		/* Waiting for the next message from master */
		MPI_Recv(&res_recv[0], n, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Termination message */
		if (res_recv[k] == 100) {
			return;
		}

		/* Solve the partial problem */
		worker_solver(res_recv, k, n);

		/* Send the finish message (set the first element to n, which is impossible for a solution) */
		res_recv[0] = n;
		MPI_Send(&res_recv[0], n, MPI_INT, 0, 101, MPI_COMM_WORLD);
	}
}
