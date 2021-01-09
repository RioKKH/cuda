#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define CHECK(call)										\
{                                                       \
	const cudaError_t error = call;                     \
	if (error != cudaSuccess)                           \
	{                                                   \
		printf("Error: %s:%d, ", __FILE__, __LINE__);   \
        printf("code:%d, reason: %s\n", error,          \
	            cudaGetErrorString(error));             \
		exit(1);                                        \
	}                                                   \
}                                                       \

                                                          
                                                          
void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n",
					hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match) {
		printf("Arrays match.\n\n");
	}

	return;
}


void initialData(float *ip, int size) {
	// 乱数シードを生成
	time_t t;
	srand((unsigned) time(&t));

	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}


void sumArraysOnHost(float *A, float *B, float *C, const int N) {
	for (int idx = 0; idx < N; idx++) {
		C[idx] = A[idx] + B[idx];
	}
}


__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}


int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// デバイスのセットアップ
	int dev = 0;
	cudaSetDevice(dev);

	// ベクトルのデータサイズを設定
	int nElem = 32;
	printf("Vector size %d\n", nElem);

	// ホストメモリを確保
	size_t nBytes = nElem * sizeof(float);
	printf("sizeof(float): %ld, sizeof(hostMemory): %ld\n",
			sizeof(float), nBytes);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// ホスト側でデータを初期化
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// デバイスのグローバルメモリを確保
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes); 
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	// ホストからデバイスへデータを転送
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

	// ホスト側でカーネルを呼び出す
	dim3 block(nElem);
	dim3 grid(1);

	sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C);
	// sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C, nElem);
	printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);

	// カーネルの結果をホスト側にコピー
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	// 結果をチェックするためにホスト側えベクトルを加算
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	// デバイスの結果とチェック
	checkResult(hostRef, gpuRef, nElem);

	// デバイスのグローバルメモリを開放
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// ホストのメモリを開放
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();
	return(0);
}

