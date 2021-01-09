#include <stdio.h>

#define WIDTH 1024 // 処理対象の行列のサイズはWIDTH x WIDTH.

// ホスト (CPU)側の行列定義
float h_A[WIDTH * WIDTH];
float h_B[WIDTH * WIDTH];
float h_C[WIDTH * WIDTH];

void h_multiply(float *A, float *B, float *C);

// メイン関数
int main()
{
	unsigned int i;

	// ホスト側の行列に値をセット
	for (i = 0; i < (WIDTH * WIDTH); i++) {
		h_A[i] = (float)i;
		h_B[i] = (float)i;
	}

	// ホスト側での計算結果
	h_multiply(h_A, h_B, h_C);
	printf("   Calculation result: %f\n", h_C[WIDTH * WIDTH - 1]);
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
				tmp += A[WIDTH * r + i] * B[WIDTH * i +c];
			}
			C[WIDTH * r + c] = tmp;
		}
	}
}

