#include <cuda_runtime.h>
#include <stdio.h>

/***
  アプリケーションのデータサイズは一定にキープ。
  ブロックサイズの変化似合わせてグリッドサイズが変化することがわかる
*/
int main(int argc, char **argv)
{
	// データ要素の合計数を定義
	int nElem = 1024;

	// グリッドとブロックの構造を定義
	dim3 block(1024);
	dim3 grid((nElem + block.x - 1) / block.x);
	// (1024 + 1024 -1) / 1024 = 2047 / 1024 = 1 --> grid = 1
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	// ブロックをリセット
	block.x = 512;
	grid.x = (nElem + block.x - 1) / block.x;
	// (1024 + 512 -1) / 512 = 1535 / 512 = 2 --> grid = 2
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	// ブロックをリセット
	block.x = 256;
	grid.x = (nElem + block.x - 1) / block.x;
	// (1024 + 256 - 1) / 256 = 1279 / 256 = 4
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	// ブロックをリセット
	block.x = 128;
	grid.x = (nElem + block.x - 1) / block.x;
	// (1024 + 128 - 1) / 128 = 1151 / 128 = 8
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	// デバイスをリセット
	cudaDeviceReset();
}
