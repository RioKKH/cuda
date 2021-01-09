#include <stdio.h>
#include <cuda_runtime.h> // 基本的に必要
#include "timer.h"

#define BLOCK 16 // 各ブロックは16x16個のスレッドから定義されるものとする
#define WIDTH 1024 // 処理対象の行列のサイズはWIDTH x WIDTH.

// ホスト (CPU)側の行列定義
float h_A[WIDTH * WIDTH];
float h_B[WIDTH * WIDTH];
float h_C[WIDTH * WIDTH];

// デバイス(GPU)側の行列へのポインタ
float *d_A, *d_B, *d_C;

void h_multiply(float *A, float *B, float *C);
__global__ void d_multiply0(float *A, float *B, float *C);
__global__ void d_multiply1(float *A, float *B, float *C);

// メイン関数
int main()
{
	unsigned int i;

	// デバイス側に行列様のメモリを確保
	cudaMalloc((void**)&d_A, sizeof(float) * WIDTH * WIDTH);
	cudaMalloc((void**)&d_B, sizeof(float) * WIDTH * WIDTH);
	cudaMalloc((void**)&d_C, sizeof(float) * WIDTH * WIDTH);

	// ホスト側の行列に値をセット
	for (i = 0; i < (WIDTH * WIDTH); i++) {
		h_A[i] = (float)i;
		h_B[i] = (float)i;
	}

    // 計算時間計測用のタイマーのセット
	StartTimer();

	// ホスト側の行列のデータをデバイス側の行列へ転送
	cudaMemcpy(d_A, h_A, sizeof(float) * WIDTH * WIDTH, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * WIDTH * WIDTH, cudaMemcpyHostToDevice);

	// グリッドとブロックの定義
	// WIDTHがBLOCKで割り切れることを前提としている。割り切れない場合には
	// 計算されない行列要素が生じてしまう
	dim3 grid(WIDTH / BLOCK, WIDTH / BLOCK);
	dim3 block(BLOCK, BLOCK);

	// GPU処理の起動
	// d_multiply0 <<< grid, block >>> (d_A, d_B, d_C);
	d_multiply1 <<< grid, block >>> (d_A, d_B, d_C);

	// 計算経過はd_Cに格納されているので、それをホスト側のh_Cへ転送
	cudaMemcpy(h_C, d_C, sizeof(float) * WIDTH * WIDTH, cudaMemcpyDeviceToHost);

	// 計算経過の表示
	printf("デバイス計算時間:    %f(ms)  ", GetTimer());
	printf("デバイス計算結果:    %f\n", h_C[WIDTH * WIDTH - 1]);

	// デバイス側のメモリを開放
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 比較様にホスト側でも計算してみる
	StartTimer();
	h_multiply(h_A, h_B, h_C);
	printf("ホスト計算時間:    %f(ms)  ", GetTimer());
	printf("ホスト計算結果:    %f\n", h_C[WIDTH * WIDTH - 1]);
}


void h_multiply(float *A, float *B, float *C)
{
	/* 2次元の行列を1次元で表現し、
	 * この1次元表記2次元行列の積を以下のアルゴリズムで計算する。
	 * C_{rc} = \sum_{i=0}^{WIDTH-1} A_{ri} \times B_{ic}
	 * */
	unsigned int r, c, i;
	float tmp;
	for (r = 0; r < WIDTH; r++) {
		for (c = 0; c < WIDTH; c++) {
			tmp = 0.0;
			for (i = 0; i < WIDTH; i++) {
				tmp += A[WIDTH * r + i] * B[WIDTH * i + c];
			}
			C[WIDTH * r + c] = tmp;
		}
	}
}

__global__ void d_multiply0(float *A, float *B, float *C)
{
	// 行列Cのr行c列の要素を決定するスレッドとして起動するようになっている
	unsigned int r = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i;
	float tmp;
	tmp = 0.0f;
	for (i = 0; i < WIDTH; i++) {
		tmp += A[WIDTH * r + i] * B[WIDTH * i + c];
	}
	C[WIDTH * r + c] = tmp;
}

__global__ void d_multiply1(float *A, float *B, float *C)
{
	// 行列Cのr行c列の要素を決定するスレッドとして起動するようになっている
	unsigned int r = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i, j;
	float tmp;

	__shared__ float s_A[BLOCK][BLOCK];
	__shared__ float s_B[BLOCK][BLOCK];
	tmp = 0.0f;
	// ブロック単位で枠をずらしながら部分行列の積を計算するforループ。
	for (i = 0; i < WIDTH; i += BLOCK) {
		//行列の一部をシェア−ドメモリに確保
		s_A[threadIdx.y][threadIdx.x] = A[WIDTH * r + i + threadIdx.x];
		s_B[threadIdx.y][threadIdx.x] = B[WIDTH * (i + threadIdx.y) + c];
		__syncthreads();
		// シェアードメモリで積を計算
		for (j = 0; j < BLOCK; j++) {
			tmp += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
		}
		__syncthreads();
	}
	C[WIDTH * r + c] = tmp;
}
