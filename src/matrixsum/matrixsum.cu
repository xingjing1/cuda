#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>

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
	double epsilon = 1.0E-8;
	bool match = 1;
	for(int i = 0; i < N; i++){
		if(abs(hostRef[i] - gpuRef[i]) > epsilon){
			match = 0;
			printf("Array do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if(match) printf("Array match.\n\n");

}

void sumArrayOnHostRow(float *A, float *B, const int row, const int col){
        for(int i = 0; i < row; i++){
		B[i] = 0;
        	for(int j = 0; j < col; j++){
			B[i] += A[j + i * col];
		}
	}
}

void sumArrayOnHostCol(float *A, float *B, const int row, const int col){
        for(int i = 0; i < col; i++){
		B[i] = 0;
        	for(int j = 0; j < row; j++){
			B[i] += A[j * col + i];
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

__global__ void sumArrayOnDeviceRow(float *A, float *B, const int row, const int col){
	int idx = blockIdx.x * blockDim.x  + threadIdx.x;
	float tmp = 0;
        for(int i = 0; i < col; i++){
		tmp += A[i + idx * col];
	}
	B[idx] = tmp;
}

__global__ void sumArrayOnDeviceCol(float *A, float *B, const int row, const int col){
	int idx = blockIdx.x * blockDim.x  + threadIdx.x;
	float tmp = 0.0f;
        for(int i = 0; i < row; i++){
		tmp += A[i * col + idx];
	}
	B[idx] = tmp;
}

int main(int argc, char **argv){
	cudaDeviceProp deviceProp;
	int dev = 0;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device : %s\n", deviceProp.name);
	int row = 8192, col = 1024;
	int nElem = row * col;
	int nBytes = row * col * sizeof(float);

	float *h_A, *h_BRow, *h_BCol;

	h_A = (float*)malloc(nBytes);
	h_BRow = (float*)malloc(row * sizeof(float));
	h_BCol = (float*)malloc(row * sizeof(float));
	initialData(h_A, nElem);

	sumArrayOnHostRow(h_A, h_BRow, row, col);
	sumArrayOnHostCol(h_A, h_BCol, col, row);

	float *d_ARow, *d_BRow;
        double timemalloc1, timemalloc2, timemalloc3, timecopy1;
        timemalloc1 = cpuSecond();
	cudaMalloc((float**)&d_ARow, nBytes);
        timemalloc3 = cpuSecond();
	cudaMalloc((float**)&d_BRow, row*sizeof(float));
        timemalloc2 = cpuSecond();
	cudaMemcpy(d_ARow, h_A, nBytes, cudaMemcpyHostToDevice);
        timecopy1 = cpuSecond();
        printf(" timemalloc1: %f ,timemalloc2: %f  timecopy1: %f\n", timemalloc3 - timemalloc1, timemalloc2 - timemalloc3, timecopy1 - timemalloc2);
	int blockSize = 512;
	dim3 block(blockSize);
	dim3 grid(16);
	//8192*1024
	double iStart = cpuSecond();
	sumArrayOnDeviceRow<<<grid, block>>>(d_ARow, d_BRow, row, col);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;



	float *d_ACol, *d_BCol;
        timemalloc1 = cpuSecond();
	cudaMalloc((float**)&d_ACol, nBytes);
        timemalloc3 = cpuSecond();
	cudaMalloc((float**)&d_BCol, row*sizeof(float));
        timemalloc2 = cpuSecond();
	cudaMemcpy(d_ACol, h_A, nBytes, cudaMemcpyHostToDevice);
        timecopy1 = cpuSecond();
        printf(" timemalloc1: %f ,timemalloc2: %f  timecopy1: %f\n", timemalloc3 - timemalloc1, timemalloc2 - timemalloc3, timecopy1 - timemalloc2);
	int blockSize1 = 512;
	dim3 block1(blockSize1);
	dim3 grid1(16);
        //1024*8192
	double iStart2 = cpuSecond();
	sumArrayOnDeviceCol<<<grid1, block1>>>(d_ACol, d_BCol, col, row);
	cudaDeviceSynchronize();
	double iElaps2 = cpuSecond() - iStart2;
	printf("8192*1024: %5.6f, 1024*8192: %5.6f\n", iElaps, iElaps2);


	float *gpuRow, *gpuCol;
	gpuRow = (float*)malloc(row * sizeof(float));
	gpuCol = (float*)malloc(row * sizeof(float));
        timemalloc1 = cpuSecond();
	cudaMemcpy(gpuRow, d_BRow, row * sizeof(float), cudaMemcpyDeviceToHost);
        timemalloc3 = cpuSecond();
	cudamemcpy(gpucol, d_bcol, row * sizeof(float), cudamemcpydevicetohost);
        timemalloc2 = cpuSecond();
        printf(" timemalloc1: %f ,timemalloc2: %f  timecopy1: %f\n", timemalloc3 - timemalloc1, timemalloc2 - timemalloc3, timecopy1 - timemalloc2);
	checkResult(h_BRow, gpuRow, row);
	checkResult(h_BCol, gpuCol, row);
	free(h_A);
	free(h_BCol);
	free(h_BRow);
	free(gpuCol);
	free(gpuRow);
	cudaFree(d_ARow);
	cudaFree(d_BRow);


	return 0;
}
