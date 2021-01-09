#include <stdio.h>
#include <iostream>

__global__ void helloFromGPU()
{
	printf("Hello world from GPU using C++\n");
	// A line below doesn't work!
	// std::cout << "Hello world from GPU using C++" << std::endl;
} 

int main(int argc, char const* argv[])
{
	std::cout << "Hello world from cpu using C++" << std::endl;
	helloFromGPU <<<1, 10>>>();
	return 0;
}
