#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>


#define TSM 64
#define TSN 64
#define TSK 2
#define WPTM 4
#define WPTN 4
#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#define TRANSPOSEX 16
#define TRANSPOSEY 16


double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp, NULL);

	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
#define CHECK(call)                    \
{                                      \
    const cudaError_t error = call;    \
    if(error != cudaSuccess)           \
    {                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);  \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit();                        \
    }                                  \
}                                      \

void checkResult(float* hostRef, float* NEONRef, const int N){
	double epsilon = 1.0E-2;
	bool match = 1;
	for(int i = 0; i < N; i++){
		if(abs(hostRef[i] - NEONRef[i]) > epsilon){
			match = 0;
			printf("Array do not match!\n");
			printf("host %5.2f NEON %5.2f at current %d\n", hostRef[i], NEONRef[i], i);
			break;
		}
	}
	if(match) printf("Array match.\n\n");

}

//col major -> row major
__global__ void tranpose(const int P, const int Q, const float* input,float* output){
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int ID0 = blockIdx.x * TRANSPOSEX + threadIdx.x;
	const int ID1 = blockIdx.y * TRANSPOSEY + threadIdx.y;
	__shared__ float buffer[TRANSPOSEX][TRANSPOSEY];

	if(ID0 < P && ID1 < Q){
		buffer[ty][tx] = input[ID1 * P + ID0];
	}
		__syncthreads();
	
	const int newID0 = blockIdx.y * TRANSPOSEY + tx;
	const int newID1 = blockIdx.x * TRANSPOSEX + ty;

	if(newID0 < Q && newID1 < P){
		output[newID1 * Q + newID0] = buffer[tx][ty];
	}
	
}

// col major
void gemmOnHost(const int M, const int N, const int K, 
                       const float*A, const float*B, float*C){
	float acc = 0;
	for(int j = 0; j < N; j++){
		for(int i = 0; i< M; i++){
			acc = 0;	
			for(int k = 0; k < K; k++){
				acc += A[i+k*M] * B[j*K + k];
			}
			C[j*M + i] = acc;
		}
	}
}

/*
//1*8 of C matrix per thread, col major
__global__ void sgemmOnDeviceOld(const int M, const int N, const int K,
		 const float* A, const float* B, float* C){
	int globalrow = blockIdx.x * blockDim.x + threadIdx.x;
	int globalcol = blockIdx.y * blockDim.y * WPT + threadIdx.y;
	int row = threadIdx.x;
	int col = threadIdx.y;
	__shared__ float Asub[Ts][Ts];
	__shared__ float Bsub[Ts][Ts];

	int numTiles = K/Ts;
	float acc[WPT] = {0};

	for(int i = 0; i < numTiles; i++){
		for(int j = 0; j < WPT; j++){
			int tilecol = Ts * i + col + j * RTS;
			int tilerow = Ts * i + row;
			Asub[col + j * RTS][row] = A[tilecol * M + globalrow];
			Bsub[col + j * RTS][row] = B[(globalcol +j * RTS) * K + tilerow];
		}
		__syncthreads();
		for(int k = 0; k < Ts; k++){
			for(int m = 0; m < WPT; m++){
				acc[m] += Asub[k][row] * Bsub[col+m*RTS][k];
			}
		}
		__syncthreads();
	}
	for(int i = 0; i < WPT; i++)
		C[(globalcol + i*RTS) * M + globalrow] = acc[i];
	
}
*/

//8*8 of C matrix per thread, col major
__global__ void sgemmOnDevicePadding(const int M, const int N, const int K,
		 const float* A, const float* B, float* C){
	int tidm = threadIdx.x;
	int tidn = threadIdx.y;
	int offsetM = blockIdx.x * TSM;
	int offsetN = blockIdx.y * TSN;
	__shared__ float Asub[TSK][TSM];
	__shared__ float Bsub[TSN][TSK];

	int numTiles = (K + TSK -1)/TSK;
	float Areg;
	float Breg[WPTN];
	float acc[WPTM][WPTN] = {0};

	for(int t = 0; t < numTiles; t++){
		for(int la = 0; la < LPTA; la++){
			int tid = tidn * RTSM + tidm;
			int id = la*RTSN*RTSM + tid;
			// row and col in 128*TSK
			int row = id%TSM;
			int col = id/TSM;
			int tiledIndex = TSK*t + col;
			if((tiledIndex < K)&&((offsetM + row)<M))
				Asub[col][row] = A[tiledIndex*M + offsetM + row];
			else
				Asub[col][row] = 0;
			if((tiledIndex < K)&&((offsetN + row)<N))
				Bsub[row][col] = B[tiledIndex*N + offsetN + row];
			else
				Bsub[row][col] = 0;
		}
		__syncthreads();
		for(int k = 0; k < TSK; k++){
			for(int wn = 0; wn < WPTN; wn++){
				int col = tidn + wn*RTSN;
				Breg[wn] = Bsub[col][k];
			}
			for(int wm = 0; wm < WPTM; wm++){
				int row = tidm + wm *RTSM;
				Areg = Asub[k][row];
				for(int wn = 0; wn < WPTN; wn++){
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}
		__syncthreads();
	}
	for(int i = 0; i < WPTM; i++){
		int globalRow = offsetM + tidm + i * RTSM;
		for(int j = 0; j < WPTN; j++){
			int globalCol = offsetN + tidn + j * RTSN;
			if((globalCol < N)&& (globalRow < M))
				C[globalCol * M + globalRow] = acc[i][j];
		}
	}
	
}

//8*8 of C matrix per thread, col major
__global__ void sgemmOnDeviceNoPadding(const int M, const int N, const int K,
		 const float* A, const float* B, float* C){
	int tidm = threadIdx.x;
	int tidn = threadIdx.y;
	int offsetM = blockIdx.x * TSM;
	int offsetN = blockIdx.y * TSN;
	__shared__ float Asub[TSK][TSM];
	__shared__ float Bsub[TSN][TSK];

	int numTiles = (K + TSK -1)/TSK;
	float Areg;
	float Breg[WPTN];
	float acc[WPTM][WPTN] = {0};

	for(int t = 0; t < numTiles; t++){
		for(int la = 0; la < LPTA; la++){
			int tid = tidn * RTSM + tidm;
			int id = la*RTSN*RTSM + tid;
			// row and col in 128*TSK
			int row = id%TSM;
			int col = id/TSM;
			int tiledIndex = TSK*t + col;
			Asub[col][row] = A[tiledIndex*M + offsetM + row];
			Bsub[row][col] = B[tiledIndex*N + offsetN + row];
		}
		__syncthreads();
		for(int k = 0; k < TSK; k++){
			for(int wn = 0; wn < WPTN; wn++){
				int col = tidn + wn*RTSN;
				Breg[wn] = Bsub[col][k];
			}
			for(int wm = 0; wm < WPTM; wm++){
				int row = tidm + wm *RTSM;
				Areg = Asub[k][row];
				for(int wn = 0; wn < WPTN; wn++){
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}
		__syncthreads();
	}
	for(int i = 0; i < WPTM; i++){
		int globalRow = offsetM + tidm + i * RTSM;
		for(int j = 0; j < WPTN; j++){
			int globalCol = offsetN + tidn + j * RTSN;
			C[globalCol * M + globalRow] = acc[i][j];
		}
	}
	
}

void initialData(float *ip, int size){
	time_t t;
	srand((unsigned int) time(&t));
	for(int i = 0; i< size;i++){
		ip[i] = float(rand()&0xFF)/1000;
	}
}


int main(int argc, char **argv){
	cudaDeviceProp deviceProp;
	int dev = 0;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device : %s\n", deviceProp.name);

	int M = 4096, N = 4096, K = 4096;
	if(argc == 4){
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		K = atoi(argv[3]);
	}
	
        float *A, *B, *C;
	A = (float*)malloc(M * K * sizeof(float));
	B = (float*)malloc(K * N * sizeof(float));
	C = (float*)malloc(M * N * sizeof(float));
	initialData(A, M * K);
	initialData(B, K * N);

//	gemmOnHost(M, N, K, A, B, C);

        float *dA, *dB, *dC; 
	float *dBtmp;
	cudaMalloc((float**)&dA, M*K*sizeof(float));
	cudaMalloc((float**)&dB, K*N*sizeof(float));
	cudaMalloc((float**)&dBtmp, M*N*sizeof(float));
	cudaMalloc((float**)&dC, M*N*sizeof(float));
	cudaMemcpy(dA, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dBtmp, B, M*N*sizeof(float), cudaMemcpyHostToDevice);


	dim3 grid1((M + TRANSPOSEX - 1)/TRANSPOSEX, (N + TRANSPOSEY - 1)/TRANSPOSEY, 1);
	dim3 block1(TRANSPOSEX, TRANSPOSEY, 1);
	dim3 block(TSM/WPTM, TSN/WPTN, 1);
	dim3 grid((M + TSM - 1)/TSM, (N + TSN - 1)/TSN, 1);

	for(int i = 0; i < 10; i++) {
	    tranpose<<<grid1, block1>>>(K, N, dBtmp, dB);
	    sgemmOnDeviceNoPadding<<<grid, block>>>(M, N, K, dA, dB, dC);
	    cudaDeviceSynchronize();
	}
        int times = 10;
	double iStart = cpuSecond();

	for(int i = 0; i < times; i++) {
	    tranpose<<<grid1, block1>>>(K, N, dBtmp, dB);
	    sgemmOnDevicePadding<<<grid, block>>>(M, N, K, dA, dB, dC);
	    cudaDeviceSynchronize();
	}
	double iElaps = (cpuSecond() - iStart);///30;


        float *gpuC;
	gpuC = (float*)malloc(M * N * sizeof(float));
        
	cudaMemcpy(gpuC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkResult(C, gpuC, M*N);
	for(int i = 0; i < M*N; i++){
	//	printf("  %5.2f,%5.2f  ", B[i], gpuB[i]);
	//	printf("  %d,%d  ", (M + TRANSPOSEX - 1)/TRANSPOSEX, (N + TRANSPOSEY - 1)/TRANSPOSEY);
	}
	double Gflops = times * ((float)M*N*K*2/1000000000)/iElaps;
	printf("times: %5.10f, Gflops: %5.2f \n", iElaps, Gflops);

	free(A);
	free(B);
	free(C);
	free(gpuC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);


	return 0;
}
