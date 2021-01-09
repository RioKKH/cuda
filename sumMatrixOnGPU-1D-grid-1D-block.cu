#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

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


void initialData(float *ip, const int size) {
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() &0xFF);
	}
	return;
}

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

void checkResult(float *hostRef, float *gpuRef, const int N) {
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (hostRef[i] != gpuRef[i]) {
			match = 0;
			printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (match) {
		printf("Arrays match.\n\n");
	} else {
		printf("Arrays do not match.\n\n");
	}
}

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
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

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC,
		                         int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

	if (ix < nx) {
		for (int iy = 0; iy < ny; iy++) {
			int idx = iy * nx + ix;
			MatC[idx] = MatA[idx] + MatB[idx];
		}
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
	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	// ホストメモリを確保
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// ホスト側でデータを初期化
	double iStart = cpuSecond();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	double iElaps = cpuSecond() - iStart;
	printf("cudaMalloc and cudaMemcpy elapsed %f sec\n", iElaps);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// 結果をチェックするためにホスト側で行列を加算
	iStart = cpuSecond();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	iElaps = cpuSecond() - iStart;
	printf("cudaMalloc and cudaMemcpy elapsed %f sec\n", iElaps);

	// デバイスのグローバルメモリを確保
	float *d_MatA, *d_MatB, *d_MatC;
	CHECK(cudaMalloc((void **)&d_MatA, nBytes));
	CHECK(cudaMalloc((void **)&d_MatB, nBytes));
	CHECK(cudaMalloc((void **)&d_MatC, nBytes));

	// ホストからデバイスへデータを転送
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// ホスト側でカーネルを呼び出す
	int dimx = 128;
	int dimy = 1;
	// int dimx = 32;
	// int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, 1);
	// dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	iStart = cpuSecond();
	sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
	CHECK(cudaDeviceSynchronize());
	iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n",
		   grid.x, grid.y, block.x, block.y, iElaps);

	// カーネルエラーをチェック
	CHECK(cudaGetLastError());

	// カーネル側の結果をホスト側にコピー
	CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
	
	// デバイスの結果をチェック
	checkResult(hostRef, gpuRef, nxy);

	// デバイスのグローバルメモリを解放
	CHECK(cudaFree(d_MatA));
	CHECK(cudaFree(d_MatB));
	CHECK(cudaFree(d_MatC));

	// ホストのメモリを解放
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	// デバイスをリセット
	CHECK(cudaDeviceReset());

	return(0);
}
