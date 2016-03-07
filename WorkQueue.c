#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define QUEUESIZE 256
#define TAGREQ 0
#define TAGANS 1
#define TAGEMP 2
#define TAGEND 3
#define TESTTIME 10

typedef struct queue
{
	int front, back;
	int *elements;
} Q, *Qptr;

Qptr QInit()
{
	Qptr q = (Qptr)malloc(sizeof(Q));
	q->elements = (int*)malloc(QUEUESIZE * sizeof(int));
	if (!q || !q->elements) { perror("Fail to allocate memory!\n");  exit(-1); }

	int i;
	for (i = 0; i < QUEUESIZE; ++i)
	{
		q->elements[i] = rand() % 1000;

		//printf("%d\n", q->elements[i]);
	}

	q->front = 0;
	q->back = QUEUESIZE - 1;

	return q;
}

void QDelete(Qptr *q)
{
	free((*q)->elements);
	free(*q);
	*q = NULL;
}

int QPop(Qptr q)
{
	if (!q) { printf("Queue has not been initialized!\n"); exit(-1); }
	if (q->front > q->back) return -1;
	int r = q->elements[q->front];
	q->elements[q->front++] = -1;

	return r;
}

void doWork(int t)
{
	usleep(t);
}

double master(int w, int world_size)
{
	double elapsed = 0;
	MPI_Status status;
	int i;
	int msgin = -1;
	int isEmpty = 0;
	int emptyMsgSent = 0;

	Qptr q = QInit();
	/*for (i = 0; i < QUEUESIZE; ++i)
	{
	printf("%d\n", QPop(q));
	}*/

	int *msgout = (int *)malloc(w*sizeof(int));
	if (!msgout) { perror("Fail to allocate memory!\n");  exit(-1); }

	while (1)
	{
		for (i = 0; i < w; ++i)
		{
			msgout[i] = QPop(q);
			//printf("%d\n", msgout[i]);
			if (msgout[i] == -1)
			{
				isEmpty = 1;
				//printf("The work queue is empty.\n");
			}
		}
		if (isEmpty == 1)
		{
			//printf("The work queue is empty.\n");
			MPI_Recv(&msgin, 1, MPI_INT, MPI_ANY_SOURCE, TAGREQ, MPI_COMM_WORLD, &status);
			//printf("Proc 0 received workreq msg from proc %d\n", status.MPI_SOURCE);
			MPI_Send(msgout, w, MPI_INT, status.MPI_SOURCE, TAGEMP, MPI_COMM_WORLD);
			//printf("Proc 0 sent empty (-1) msg to proc %d\n", status.MPI_SOURCE);

			emptyMsgSent++;

			if (emptyMsgSent == world_size - 1)
			{
				for (i = 0; i < w; ++i)
				{
					msgout[i] = -2;
				}

				for (i = 1; i < world_size; ++i)
				{
					MPI_Send(msgout, w, MPI_INT, i, TAGEND, MPI_COMM_WORLD);
					//printf("Proc 0 sent end (-2) msg to proc %d\n", i);
				}

				elapsed += MPI_Wtime();
				break;
			}
		}
		else
		{
			MPI_Recv(&msgin, 1, MPI_INT, MPI_ANY_SOURCE, TAGREQ, MPI_COMM_WORLD, &status);
			if (elapsed == 0)
				elapsed = -MPI_Wtime();
			//printf("Proc 0 received workreq msg from proc %d\n", status.MPI_SOURCE);
			MPI_Send(msgout, w, MPI_INT, status.MPI_SOURCE, TAGANS, MPI_COMM_WORLD);
			//printf("Proc 0 sent work msg to proc %d\n", status.MPI_SOURCE);
		}
	}

	free(msgout);
	QDelete(&q);

	return 	elapsed;
}

double slaves(int w, int world_size, int world_rank)
{
	double elapsed = -MPI_Wtime();;
	MPI_Status status;
	int i;
	int msgout = 1;
	int *msgin = (int *)malloc(w*sizeof(int));
	if (!msgin) { perror("Fail to allocate memory!\n");  exit(-1); }
	//printf("Inside proc %d\n", world_rank);
	while (1)
	{
		MPI_Send(&msgout, 1, MPI_INT, 0, TAGREQ, MPI_COMM_WORLD);
		//printf("Proc %d sent workreq msg to proc 0\n", world_rank);
		MPI_Recv(msgin, w, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if (status.MPI_TAG == TAGANS)
		{
			//printf("Proc %d received work msg from proc 0\n", world_rank);
			for (i = 0; i < w; ++i)
			{
				//printf("%d\n", msgin[i]);
				doWork(msgin[i]);
			}
		}
		else if (status.MPI_TAG == TAGEMP)
		{
			//printf("Proc %d received empty msg (-1) from proc 0 - no more work\n", world_rank);
			elapsed += MPI_Wtime();
			break;
		}
	}

	MPI_Recv(msgin, w, MPI_INT, 0, TAGEND, MPI_COMM_WORLD, &status);
	//printf("Proc %d received end msg (-2) from proc 0 - all done\n", world_rank);

	free(msgin);
	return elapsed;
}

int main(int argc, char** argv) {
	int w = 1;
	if (argc == 2)
	{
		//printf("%d\n", atoi(argv[1]));
		w = atoi(argv[1]);
	}

	double timer = 0;
	int i;

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	for (i = 0; i < TESTTIME; ++i)
	{
		if (world_rank == 0)
		{
			timer += master(w, world_size);
		}
		else
		{
			timer += slaves(w, world_size, world_rank);
		}
	}

	printf("Avg. time (10) proc %d used: %.17g\n", world_rank, timer/TESTTIME);

	MPI_Finalize();
}