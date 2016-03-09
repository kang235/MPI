#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <Windows.h>

#define N 16

int world_size = -1, world_rank = -1, stride = -1;

void loadMatrices(int ***A, int ***TB, int ***C)
{
	int i, j;

	//load (initialize) matrix A
	*A = (int **)malloc(N*sizeof(int *) + N*N*sizeof(int));
	if (!(*A)) { perror("Failed malloc memory for matrix A\n"); exit(-1); }

	for (i = 0; i < N; ++i)
	{
		(*A)[i] = (int *)&(*A)[N] + i * N;
	}

	printf("Matrix A (%dX%d):\n", N, N);
	for (i = 0; i < N; ++i) 
	{
		for (j = 0; j < N; ++j)
		{
			(*A)[i][j] = rand() % 10;
			printf("%d ", (*A)[i][j]);
		}
		printf("\n");
	}

	//load (initialize) matrix B
	int **B = (int **)malloc(N*sizeof(int *) + N*N*sizeof(int));
	if (!B) { perror("Failed malloc memory for matrix B\n"); exit(-1); }

	for (i = 0; i < N; ++i)
	{
		B[i] = (int *)&B[N] + i * N;
	}

	printf("\nMatrix B (%dX%d):\n", N, N);
	for (i = 0; i < N; ++i) 
	{
		for (j = 0; j < N; ++j)
		{
			B[i][j] = rand() % 10;
			printf("%d ", B[i][j]);
		}
		printf("\n");
	}

	//transpose B to TB
	*TB = (int **)malloc(N*sizeof(int *) + N*N*sizeof(int));
	if (!(*TB)) { perror("Failed malloc memory for matrix T(B)\n"); exit(-1); }

	for (i = 0; i < N; ++i)
	{
		(*TB)[i] = (int *)&(*TB)[N] + i * N;
	}

	printf("\nMatrix T(B) (%dX%d):\n", N, N);
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			(*TB)[i][j] = B[j][i];
			printf("%d ", (*TB)[i][j]);
		}
		printf("\n");
	}

	free(B);

	//create result Matrix C
	*C = (int **)malloc(N*sizeof(int *) + N*N*sizeof(int));
	if (!(*C)) { perror("Failed malloc memory for matrix C\n"); exit(-1); }

	for (i = 0; i < N; ++i)
	{
		(*C)[i] = (int *)&(*C)[N] + i * N;
	}
}

void unloadMatrices(int ***A, int ***B, int ***C)
{
	free(*A);
	free(*B);
	free(*C);
}

int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank == 0)  assert( N % world_size == 0);

	stride = N / world_size;

	// Matrix A and Matrix T(B) and result Matrix C
	int **A = NULL, **TB = NULL, **C = NULL;
	int *sendA = NULL, *sendB = NULL;
	int *recvA = (int *)malloc(sizeof(int)*N*stride);
	int *recvB = (int *)malloc(sizeof(int)*N*stride);
	int *recvTmp = (int *)malloc(sizeof(int)*N*stride);
	if (!recvA || !recvB || !recvTmp) { perror("Failed malloc memory for recv buffer\n"); exit(-1); }

	if (world_rank == 0)
	{
		loadMatrices(&A, &TB, &C);
		sendA = A[world_rank];
		sendB = TB[world_rank];
	}

	// Distribute the rows of A across the P processors using a single collective communication command
	MPI_Scatter(sendA, N*stride, MPI_INT, recvA, N*stride, MPI_INT, 0, MPI_COMM_WORLD);

	// Distribute the cols of B across the P processors using a single collective communication command
	MPI_Scatter(sendB, N*stride, MPI_INT, recvB, N*stride, MPI_INT, 0, MPI_COMM_WORLD);

	int *partialRes = (int *)calloc(N*stride, sizeof(int));
	if (!partialRes) { perror("Failed malloc memory for partial result buffer\n"); exit(-1); }

	int shiftTimes = N / stride;
	int counter = 0;
	int des = world_rank - 1;
	if (des == -1) des = world_size - 1;
	int src = world_rank + 1;
	if (src == world_size) src = 0;

	int i, j, m, n;

	do
	{
		for (i = 0; i < stride; ++i)
		{
			for (j = 0; j < stride; ++j)
			{
				for (m = i*N, n = j*N; m < (i + 1)*N; ++m, ++n)
				{
					int index0 = j + world_rank*stride + counter*stride;
					if (index0 >= N) index0 -= N;
					int index1 = i*N;

					// Form the corresponding elements of the result of C
					partialRes[index0 + index1] += recvA[m] * recvB[n];
				}

			}
		}

		if (counter == shiftTimes - 1) break;
		else counter++;

		// The columns on each processor will be shifted to an adjacent processor
		if (world_rank == 0)
		{
			MPI_Ssend(recvB, N*stride, MPI_INT, des, 0, MPI_COMM_WORLD);
			MPI_Recv(recvTmp, N*stride, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
			MPI_Recv(recvTmp, N*stride, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Ssend(recvB, N*stride, MPI_INT, des, 0, MPI_COMM_WORLD);
		}

		int *tmp = recvTmp;
		recvTmp = recvB;
		recvB = tmp;
	} while (1);

	// Bring all the C matrix results on different processor to processor 0
	MPI_Gather(partialRes, N*stride, MPI_INT, &C[N], N*stride, MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0)
	{
		printf("\nMatrix C (%dX%d):\n", N, N);
		for (i = 0; i < N; ++i)
		{
			for (j = 0; j < N; ++j)
			{
				printf("%d ", C[i][j]);

			}

			printf("\n");
		}
	}

	free(partialRes);
	free(recvB);
	free(recvA);
	if (world_rank == 0)
	{
		unloadMatrices(&A, &TB, &C);
	}

	MPI_Finalize();
}