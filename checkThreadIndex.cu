#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                          \
{                                                                            \
	const cudaError_t error = call;                                          \
	if (error != cudaSuccess)                                                \
	{                                                                        \
		printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
		exit(1);                                                             \
	}                                                                        \
}                                                                            \

void printMatrix(int *C, const int nx, const int ny) {
	int *ic = C;
	printf("\nMatrix: (%d.%d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
	return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
		   "global index %2d ival %2d\n",
		   threadIdx.x, threadIdx.y,
		   blockIdx.x, blockIdx.y,
		   ix, iy, idx, A[idx]);
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
	ia += nx;
	ib += nx;
	ic += nx;
	}
	return;
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC,
		                         int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}


int main(int argc, char **argv) {
	printf("%s Starting...\n", argv[0]);

	// デバイス情報を取得
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// 行列の次元を設定
	int nx = 8;
	int ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	// ホストメモリを確保
	int *h_A;
	h_A = (int *)malloc(nBytes);

	// ホスト行列を整数で初期化
	for (int i = 0; i < nxy; i++) {
		h_A[i] = i;
	}
	printMatrix(h_A, nx, ny);

	// デバイスメモリを確保
	int *d_MatA;
	CHECK(cudaMalloc((void **)&d_MatA, nBytes));

	// ホストからデバイスへデータを転送
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

	// 実行設定をセットアップ
	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// カーネルを呼び出す
	printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
	CHECK(cudaDeviceSynchronize());

	// ホストとデバイスのメモリを解放
	CHECK(cudaFree(d_MatA));
	free(h_A);


	// デバイスをリセット
	CHECK(cudaDeviceReset());

	return(0);
}



