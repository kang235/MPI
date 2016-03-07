#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define ARRAYSIZE 1000000
#define TESTTIME 10
#define LOWERBOUND(p, P, n) (p*n/P)
#define UPPERBOUND(p, P, n) ((p+1)*n/P) -1
#define BLOCKSIZE(p, P, n) ((p+1)*n/P) - (p*n/P)
#define OWNER(i, P, n) (P*(i+1)-1)/n 

double sequential_sum(int world_rank, int *a, int count)
{
	if (world_rank != 0) return 0;

	assert(count <= ARRAYSIZE);

	double elapsed = -MPI_Wtime();
	int i, sum = 0;
	for (i = 0; i < count; ++i)
		sum += a[i];
	//printf("Sum: %d\n", sum);
	elapsed += MPI_Wtime();
	printf("Time used in sequential method for %d elements: %f\n", count, elapsed);

	return elapsed;
}

int psum(int world_rank, int world_size, int *a, int count)
{
	int i;
	int l = LOWERBOUND(world_rank, world_size, count);
	int u = UPPERBOUND(world_rank, world_size, count);
	int sump = 0;

	for (i = l; i <= u; ++i)
	{
		sump += a[i];
	}

	return sump;
}


double fan_in_tree(int world_rank, int world_size, int *a, int count)
{
	assert(count <= ARRAYSIZE);

	MPI_Barrier(MPI_COMM_WORLD);
	double elapsed = -MPI_Wtime();
	int i;
	int sump = psum(world_rank, world_size, a, count);

	if (world_rank == 0)
	{
		for (i = 1; i < world_size; ++i)
		{
			int inmsg = 0;
			int rc = MPI_Recv(&inmsg, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (rc != MPI_SUCCESS) { printf("Fail to receive msg to proc 0!\n");  exit(-1); }

			sump += inmsg;
		}

		//printf("Sum: %d\n", sump);
	}
	else
	{
		int rc = MPI_Send(&sump, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		if (rc != MPI_SUCCESS) { printf("Fail to send msg from proc %d!\n", world_rank);  exit(-1); }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	elapsed += MPI_Wtime();

	if (world_rank == 0)
	{
		printf("Time used in send/recv method for %d elements: %f\n", count, elapsed);
	}

	return elapsed;
}

double collective_reduce(int world_rank, int world_size, int *a, int count)
{
	assert(count <= ARRAYSIZE);

	MPI_Barrier(MPI_COMM_WORLD);
	double elapsed = -MPI_Wtime();
	int sump = psum(world_rank, world_size, a, count);
	int sum = 0;

	MPI_Reduce(&sump, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	/*if (world_rank == 0)
	{
	printf("Sum: %d\n", sum);
	}*/

	MPI_Barrier(MPI_COMM_WORLD);
	elapsed += MPI_Wtime();
	if (world_rank == 0)
	{
		printf("Time used in collective reduce method for %d elements: %f\n", count, elapsed);
	}

	return elapsed;
}

int main(int argc, char** argv) {
	int *a = (int*)malloc(ARRAYSIZE * sizeof(int));
	if (!a) { perror("Fail to allocate memory!\n");  exit(-1); }

	int i;
	//int sum = 0;
	//this should run on every proc, and rand() should give same results
	for (i = 0; i < ARRAYSIZE; ++i)
	{
		a[i] = rand() % 10;
		//sum += a[i];
	}
	//printf("%d\n", sum);

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	//double seqt[3] = { 0 };
	double fant[3] = { 0 };
	double colt[3] = { 0 };

	for (i = 0; i < TESTTIME; ++i)
	{
		/*seqt[0] += sequential_sum(world_rank, a, 1000000);
		seqt[1] += sequential_sum(world_rank, a, 100000);
		seqt[2] += sequential_sum(world_rank, a, 10000);*/

		fant[0] += fan_in_tree(world_rank, world_size, a, 1000000);
		fant[1] += fan_in_tree(world_rank, world_size, a, 100000);
		fant[2] += fan_in_tree(world_rank, world_size, a, 10000);

		colt[0] += collective_reduce(world_rank, world_size, a, 1000000);
		colt[1] += collective_reduce(world_rank, world_size, a, 100000);
		colt[2] += collective_reduce(world_rank, world_size, a, 10000);
	}

	if (world_rank == 0)
	{
		printf("\nTest time: %d\n", TESTTIME);

		//printf("Avg. time used in sequential method for 1000000 elements:\n  %.17g\n", seqt[0] / TESTTIME);
		//printf("Avg. time used in sequential method for 100000 elements:\n  %.17g\n", seqt[1] / TESTTIME);
		//printf("Avg. time used in sequential method for 10000 elements:\n  %.17g\n", seqt[2] / TESTTIME);

		printf("Avg. time used in send/recv method for 1000000 elements:\n  %.17g\n", fant[0] / TESTTIME);
		printf("Avg. time used in send/recv method for 100000 elements:\n  %.17g\n", fant[1] / TESTTIME);
		printf("Avg. time used in send/recv method for 10000 elements:\n  %.17g\n", fant[2] / TESTTIME);

		printf("Avg. time used in collective reduce method for 1000000 elements:\n  %.17g\n", colt[0] / TESTTIME);
		printf("Avg. time used in collective reduce method for 100000 elements:\n  %.17g\n", colt[1] / TESTTIME);
		printf("Avg. time used in collective reduce method for 10000 elements:\n  %.17g\n", colt[2] / TESTTIME);
	}

	MPI_Finalize();
	free(a);
}