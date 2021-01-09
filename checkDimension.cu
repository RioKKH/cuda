#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d %d)"
		   "blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
		   threadIdx.x, threadIdx.y, threadIdx.z,
		   blockIdx.x, blockIdx.y, blockIdx.z,
		   blockDim.x, blockDim.y, blockDim.z,
		   gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
	// データ要素の合計数を定義
	int nElem = 6;
	// グリッドとブロックの構造を定義
	// ブロックサイズ(スレッド数)は3
	dim3 block(3);
	// グリッドのサイズをブロックのサイズの倍数に切り上げる
	// (6 + (3-1)) / 3 = 8 / 3 = 2 
	// つまりgrid sizeは2となる。
	dim3 grid((nElem + block.x - 1) / block.x);


	// グリッドとブロックのサイズをデバイス側からチェック
	checkIndex<<<grid, block>>>();

	// デバイスをリセット
	cudaDeviceReset();

	return(0);
}

