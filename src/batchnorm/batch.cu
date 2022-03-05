#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>


#define TSM 128
#define TSN 128
#define TSK 4
#define WPTM 8
#define WPTN 8
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

void checkResult(float* hostRef, float* gpuRef, const int N){
	double epsilon = 1.0E-4;
	bool match = 1;
	for(int i = 0; i < N; i++){
		if(abs(hostRef[i] - gpuRef[i]) > epsilon){
			match = 0;
			printf("Array do not match!\n");
			printf("host %f gpu %f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if(match) printf("Array match.\n\n");

}

// col major
void batchnormOnHost(int M, int N, float* A, float* B, float* BT, float* C, float* scale){
	float tmp = 0;
        for(int i = 0; i < M; i++){
		tmp = 0;
		for(int j = 0; j < N; j++){
			tmp += A[j * M  + i];
		}
		B[i] = tmp / N;
	}

	float tmp2, tmp3;
        for(int i = 0; i < M; i++){
		tmp3 = 0;
		for(int j = 0; j < N; j++){
			tmp2 = A[j * M  + i];  
			tmp2 -= B[i];
			C[j * M + i] = tmp2 * scale[j];
			tmp2 = tmp2 * tmp2;
			tmp3 += tmp2;

		}

		tmp3 = tmp3 / N;
		tmp3 = sqrt(tmp3) + 1e-64;
		for(int j = 0; j < N; j++){
			C[j * M + i] /= tmp3;
		}

	}

	
}


__global__ void batchnormOnDevice(const int M, const int N, float* A, float* B, float* BT, float* C, float* scale){
	int idx = blockIdx.x * blockDim.x  + threadIdx.x;
	float tmp = 0;
        for(int i = 0; i < N; i++){
		tmp += A[i * M  + idx];
	}
	B[idx] = tmp / N;
	//mean

	float tmp2 = 0, tmp3 = 0;
        for(int i = 0; i < N; i++){
		tmp2 = A[i * M  + idx];  

		tmp2 -= B[idx];
		C[i * M + idx] = tmp2 * scale[i];


		tmp2 = tmp2 * tmp2;
		tmp3 += tmp2;
        }
	tmp3 = tmp3 / N;
	tmp3 = sqrt(tmp3) + 1e-64;

	for(int i = 0; i < N; i++){
		C[i * M + idx] /= tmp3;
	}
}

void initialData(float *ip, int size){
	time_t t;
	srand((unsigned int) time(&t));
	for(int i = 0; i< size;i++){
		ip[i] = float(rand()&0xFF)/10;
	}
}


int main(int argc, char **argv){
	cudaDeviceProp deviceProp;
	int dev = 0;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device : %s\n", deviceProp.name);

	int M = 512, N = 1024;
	
        float *A, *B, *BT, *C, *scale;
	A = (float*)malloc(M * N * sizeof(float));
	B = (float*)malloc(M * sizeof(float));
	BT = (float*)malloc(M * sizeof(float));
	C = (float*)malloc(M * N * sizeof(float));
	scale = (float*)malloc(N * sizeof(float));
	initialData(A, M * N);
	initialData(scale, N);

	batchnormOnHost(M, N, A, B, BT, C, scale);

        float *dA, *dB, *dBT, *dC, *dscale; 
	cudaMalloc((float**)&dA, M*N*sizeof(float));
	cudaMalloc((float**)&dB, M*sizeof(float));
	cudaMalloc((float**)&dBT, M*sizeof(float));
	cudaMalloc((float**)&dC, M*N*sizeof(float));
	cudaMalloc((float**)&dscale, N*sizeof(float));
	cudaMemcpy(dA, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dscale, scale, N*sizeof(float), cudaMemcpyHostToDevice);


	dim3 grid(512/32);
	dim3 block(32);


	double iStart = cpuSecond();
	batchnormOnDevice<<<grid, block>>>(M, N, dA, dB, dBT, dC, dscale);

	cudaDeviceSynchronize();
	double iElaps = (cpuSecond() - iStart);///30;


        float *gpuC, *gpuB;
	gpuC = (float*)malloc(M * N * sizeof(float));
	gpuB = (float*)malloc(M * sizeof(float));
        
	cudaMemcpy(gpuC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpuB, dB, M * sizeof(float), cudaMemcpyDeviceToHost);
	checkResult(C, gpuC, M*N);
	printf(" time: %f \n", iElaps);
	for(int i = 0; i < M; i++){
	//	printf(" gpuB: %f ", gpuB[i]);
		
	}
	for(int i = 0; i < M*N; i++){
	//	printf(" C: %f gpuC: %f ", C[i], gpuC[i]);
	}

	free(A);
	free(B);
	free(BT);
	free(C);
	free(scale);
	free(gpuC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dBT);
	cudaFree(dC);
	cudaFree(dscale);


	return 0;
}
