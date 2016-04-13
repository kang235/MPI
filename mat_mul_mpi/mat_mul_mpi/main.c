#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define M 100
#define N 10000
#define K 100

#define TAG_ROW_A 1
#define TAG_COL_B 2
#define TAG_INDEX 3
#define TAG_RESULT 4

int world_size, world_rank;

void loadMatrices(int ***A, int ***B, int ***C)
{
	int i, j;

	//load (initialize) matrix A
	*A = (int **)malloc(M*sizeof(int *) + M*N*sizeof(int));
	if (!(*A)) { perror("Failed malloc memory for matrix A\n"); exit(-1); }

	for (i = 0; i < M; ++i)
	{
		(*A)[i] = (int *)&(*A)[M] + i * N;
	}

	//printf("Matrix A (%dX%d):\n", M, N);
	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			(*A)[i][j] = rand() % 10;
			//printf("%d ", (*A)[i][j]);
		}
		//printf("\n");
	}

	//load (initialize) matrix B
	*B = (int **)malloc(N*sizeof(int *) + N*K*sizeof(int));
	if (!(*B)) { perror("Failed malloc memory for matrix B\n"); exit(-1); }

	for (i = 0; i < N; ++i)
	{
		(*B)[i] = (int *)&(*B)[N] + i * K;
	}

	//printf("\nMatrix B (%dX%d):\n", N, K);
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j < K; ++j)
		{
			(*B)[i][j] = rand() % 10;
			//printf("%d ", (*B)[i][j]);
		}
		//printf("\n");
	}

	//create result Matrix C
	*C = (int **)malloc(M*sizeof(int *) + M*K*sizeof(int));
	if (!(*C)) { perror("Failed malloc memory for matrix C\n"); exit(-1); }

	for (i = 0; i < M; ++i)
	{
		(*C)[i] = (int *)&(*C)[M] + i * K;
	}
}

void unloadMatrices(int **A, int **B, int **C)
{
	free(A);
	free(B);
	free(C);
}

void seq_mul(const int** A, const int** B, int** C, const int m, const int n, const int k)
{
	int i, j, l;

	printf("\nResult Matrix C (%dX%d):\n", M, K);
	for (i = 0; i < m; ++i)
	{
		for (j = 0; j < k; ++j)
		{
			int sum = 0;
			for (l = 0; l < n; ++l)
			{
				sum += A[i][l] * B[l][j];
			}
			C[i][j] = sum;
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}
}

void master()
{
	int **A, **B, **C;

	loadMatrices(&A, &B, &C);

	if (M < world_size || K < world_size)
	{
		seq_mul(A, B, C, M, N, K);
	}
	else
	{
		int index = 0;
		int max = M * K;
		MPI_Status status;
		int end_sent = 0;

		while(1)
		{
			int row = index / K;
			int col = index % K;

			int index_in = -1;
			int result_in = -1;

			MPI_Recv(&index_in, 1, MPI_INT, MPI_ANY_SOURCE, TAG_INDEX, MPI_COMM_WORLD, &status);
			MPI_Recv(&result_in, 1, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

			if (index_in != -1)
				C[index_in / K][index_in % K] = result_in;

			if (index == max)
			{
				index = -1;
				MPI_Ssend(&index, 1, MPI_INT, status.MPI_SOURCE, TAG_INDEX, MPI_COMM_WORLD);
				end_sent++;
				if (end_sent == world_size - 1) break;
			} 
			else
			{
				MPI_Ssend(&index, 1, MPI_INT, status.MPI_SOURCE, TAG_INDEX, MPI_COMM_WORLD);
				MPI_Ssend(A[row], N, MPI_INT, status.MPI_SOURCE, TAG_ROW_A, MPI_COMM_WORLD);

				MPI_Datatype coltype;
				MPI_Type_vector(N, 1, K, MPI_INT, &coltype);
				MPI_Type_commit(&coltype);
				MPI_Ssend(&B[0][col], 1, coltype, status.MPI_SOURCE, TAG_COL_B, MPI_COMM_WORLD);
			}

			index++;
		}
	}

	//int i, j;
	//printf("\nResult Matrix C (%dX%d):\n", M, K);
	//for (i = 0; i < M; ++i) {
	//	for (j = 0; j < K; ++j)
	//	{
	//		printf("%d ", C[i][j]);
	//	}
	//	printf("\n");
	//}

	//seq_mul(A, B, C, M, N, K);

	unloadMatrices(A, B, C);
}

void slaves()
{
	int rowSize = M / world_size;
	int colSize = K / world_size;

	int **rowBlock = (int **)malloc(rowSize*sizeof(int *) + rowSize*N*sizeof(int));
	int **colBlock = (int **)malloc(N*sizeof(int *) + N*colSize*sizeof(int));
	int **resultBlock = (int **)malloc(rowSize*sizeof(int *) + rowSize*colSize*sizeof(int));
	int index = -1;
	int result = 0;
	int row[N], col[N];

	MPI_Status status;
	int i;
	while (1)
	{
		MPI_Ssend(&index, 1, MPI_INT, 0, TAG_INDEX, MPI_COMM_WORLD);
		MPI_Ssend(&result, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
		result = 0;

		MPI_Recv(&index, 1, MPI_INT, 0, TAG_INDEX, MPI_COMM_WORLD, &status);

		if (index == -1)
			break;

		MPI_Recv(row, N, MPI_INT, 0, TAG_ROW_A, MPI_COMM_WORLD, &status);
		MPI_Recv(col, N, MPI_INT, 0, TAG_COL_B, MPI_COMM_WORLD, &status);

		for (i = 0; i < N; ++i)
			result += row[i] * col[i];
	} 

	free(rowBlock);
	free(colBlock);
	free(resultBlock);

}

int main(int argc, char** argv) {
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank == 0)
	{
		double elapsed = -MPI_Wtime();
		master();
		elapsed += MPI_Wtime();

		printf("Time proc %d used: %.17g\n", world_rank, elapsed);
	}
	else
	{
		slaves();
	}

	MPI_Finalize();

}