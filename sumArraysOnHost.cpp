#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
        // printf("%f = %f + %f\n", C[idx], A[idx], B[idx]);
    }
}

void initialData(float *ip, int size) {
    // generate a seed for randum number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
    return ;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {
    int nElem = 1024;
    double iStart, iElaps;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    iElaps = cpuSecond() - iStart;

    free(h_A);
    free(h_B);
    free(h_C);

    return(0);
}
