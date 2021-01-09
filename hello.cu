#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU()
{
	printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main(int argc, char **argv)
{
	printf("Hello from CPU\n");

	helloFromGPU <<<1, 10>>>();
	// cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
