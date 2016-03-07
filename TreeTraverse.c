#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define ISPOWEROFTWO(x) ((x != 0) && !(x & (x - 1)))

//#define LOWERBOUND(p, P, n) (p*n/P)
//#define UPPERBOUND(p, P, n) ((p+1)*n/P) -1
//#define BLOCKSIZE(p, P, n) ((p+1)*n/P) - (p*n/P)
//#define OWNER(i, P, n) (P*(i+1)-1)/n 

int world_rank = -1, world_size = -1, extralvls = -1;

typedef struct node {
	float val;
	int lvl;
	struct node *left, *right;
} Node, *NodePtr;

NodePtr create_tree_seq(const int maxlvl, int curlvl, const int nodesnum, int *count, NodePtr *ptr, int *index)
{
	if (curlvl >= maxlvl)
	{
		return NULL;
	}
	else
	{
		if ((*count) >= nodesnum)
		{
			return NULL;
		}

		NodePtr pnode = (NodePtr)malloc(sizeof(Node));

		if (!pnode) { perror("Failed to malloc memory!\n"); return NULL; }

		pnode->val = (float)rand() / RAND_MAX; //rand a 0-1 number
		pnode->lvl = curlvl;

		if (curlvl == extralvls)
		{
			if (*index == world_rank)
			{
				//traverse start point
				*ptr = pnode;
			}
			(*index)++;
		}
			
		(*count)++;
		curlvl++;
		//printf("%f\n", pnode->val);

		pnode->left = create_tree_seq(maxlvl, curlvl, nodesnum, count, ptr, index);
		pnode->right = create_tree_seq(maxlvl, curlvl, nodesnum, count, ptr, index);

		return pnode;
	}
}

void traverse_tree_seq(const NodePtr node, int *count, const int lvl)
{
	if (lvl != -1 && node->lvl == lvl) return;

	if (node->left) traverse_tree_seq(node->left, count, lvl);
	if (node->right) traverse_tree_seq(node->right, count, lvl);

	//printf("reached end\n");
	if (node->val < 0.5)
		(*count)++;
}

void destroy_tree(NodePtr *node)
{
	NodePtr left = (*node)->left;
	NodePtr right = (*node)->right;
	if (left) destroy_tree(&left);
	if (right) destroy_tree(&right);

	free(*node);
}

//Get two values based on the number of nodes:
//fulllvl: the number of levels that are full
//remainer: there might be some remaining nodes in the final level
//			that cannot form a full level. The number of these nodes.
void get_tree_info(int nodenum, int * fulllvl, int * remainer)
{
	assert(nodenum > 0);
	*fulllvl = (int)log2((double)nodenum + 1);
	*remainer = (nodenum + 1) % (int)pow(2, *fulllvl);
}

int main(int argc, char** argv) {
	int nodesnum = 0x40000;
	//int nodesnum = 16;

	int fulllvl, remainer;
	get_tree_info(nodesnum, &fulllvl, &remainer);

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	if (world_rank == 0) assert(ISPOWEROFTWO(world_size));

	//first few levels proc 0 needs to take care of 
	extralvls = (int)log2((double)world_size) + 1;

	int counter0 = 0;
	NodePtr ptr;
	int index = 0;

	NodePtr tree = create_tree_seq(fulllvl + 1, 0, nodesnum, &counter0, &ptr, &index);
	//printf("Proc %d: tree created with %d nodes\n", world_rank, counter0);

	int counter1 = 0;

	MPI_Barrier(MPI_COMM_WORLD);
	double elapsed = -MPI_Wtime();

	traverse_tree_seq(ptr, &counter1, -1);

	if (world_rank == 0)
	{
		traverse_tree_seq(tree, &counter1, extralvls);
	}

	//printf("Proc %d: There %d nodes has val less than 0.5\n", world_rank, counter1);

	int sum = 0;
	MPI_Reduce(&counter1, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
	elapsed += MPI_Wtime();
	
	if (world_rank == 0)
	{
		//printf("There are %d nodes has val less than 0.5\n", sum);
		printf("Time spent: %.17g\n", elapsed);
	}

	destroy_tree(&tree);
	MPI_Finalize();
}