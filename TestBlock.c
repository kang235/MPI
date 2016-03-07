#include <mpi.h>
#include <stdio.h>

#define ARRAY_SIZE 50
#define BLOCK_SIZE 1

int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	int a[ARRAY_SIZE] = { 0 };

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int l, u, s;
	l = world_rank;
	s = world_size * BLOCK_SIZE;

	int i;
	for (i = l; i < ARRAY_SIZE; i += s)
	{
		//printf("%d: index %d\n", world_rank, i);
	}

	u = i - s;

	printf("%d: [%d:%d:%d]\n", world_rank, l, u, s);

	// Finalize the MPI environment.
	MPI_Finalize();
}