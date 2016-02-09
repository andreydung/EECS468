#include <stdio.h>
#include <stdlib.h>

#define NX 8

int main()
{

	int numThreads = NX*NX;

	// print the threads
	for( int i = 0; i < 32; i++ )
	{
		printf(" %i\t", i );
	}
	printf("\n");

	for( int i = 0; i < NX; i++ )
	{
		int addr1[numThreads];
		int addr2[numThreads];

		int addrB[32];
		int numB[32];


		for( int kk = 0; kk < 32; kk ++ )
		{
			addrB[kk] = -1;
			numB[kk] = 0;
		}

		for( int j = 0; j < numThreads; j++ )
		{
			int x = j/NX; 
			int y = j%NX;

			addr1[j] = y*(NX+1) + i;
			addr2[j] = (NX + 1)*(NX) + i*(NX+1) + x;
			//addr2[j] = NX*NX + x*(NX+1) + i;
			int baddr = addr1[j] % 32;
			if( addrB[baddr] == -1 )
			{
				addrB[baddr] = addr1[j];
			}
			else if( addrB[baddr] != addr1[j] )
			{
				numB[baddr]++;
			}


			//printf("%i [%i] ", addr1[j], addr1[j]%32);
		}


		for(int kk = 0; kk < 32; kk++ )
		{
			printf(" [%i]\t", numB[kk]);
			addrB[kk] = -1;
			numB[kk] = 0;
		}
	
		
		printf("\n");
		for( int j = 0; j < numThreads; j++ )
		{
			//printf("%i [%i] ", addr2[j], addr2[j]%32);

			int baddr = addr2[j] % 32;
			if( addrB[baddr] == -1 )
			{
				addrB[baddr] = addr2[j];
			}
			else if( addrB[baddr] != addr2[j] )
			{
				numB[baddr]++;
			}


		}

		for(int kk = 0; kk < 32; kk++ )
		{
			printf(" [%i]\t", numB[kk]);
		}
		
		printf("\n");
	}
	
	return 0;
}
