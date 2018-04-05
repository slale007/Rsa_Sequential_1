#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cudaMpir.h"
#include "device_launch_parameters.h"

#define LONGINT unsigned long long int


__global__ void cuda_RightShiftsBlocks(LONGINT* result, LONGINT* inputNumber, int inputNumberLength, int shift) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < inputNumberLength) {
		int target = idx - shift;
		if (target >= 0 ) {
			result[target] = inputNumber[idx];
		}
	}
}

void RightShiftBlocks(mpz_t result, mpz_t inputNumber, int shift) {
	cudaError_t cudaStatus;

	if (inputNumber->_mp_size <= shift) {
		result->_mp_size = 1;
		result->_mp_alloc = 1;
		result->_mp_d = (unsigned long long int *)malloc(sizeof(unsigned long long int));
		result->_mp_d[0] = 0;
		return;
	}

	result->_mp_size = inputNumber->_mp_size - shift;
	result->_mp_alloc = result->_mp_size;
	result->_mp_d = (unsigned long long int *)malloc(result->_mp_size * sizeof(unsigned long long int));
	LONGINT* dev_inputNumber;
	LONGINT* dev_result;

	cudaStatus = cudaMalloc((void**)&dev_inputNumber, inputNumber->_mp_size * sizeof(unsigned long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed0!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_result, result->_mp_size * sizeof(unsigned long long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed1!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_inputNumber, inputNumber->_mp_d, inputNumber->_mp_size * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed2!");
		goto Error;
	}

	int inputNumberLength = inputNumber->_mp_size;

	// cudaStatus = cudaMemset(dev_result, 0, result->_mp_size * sizeof(unsigned long long int));

	// Launch a kernel on the GPU with one thread for each element.
	cuda_RightShiftsBlocks<<<1, 64 >>>(dev_result, dev_inputNumber, inputNumberLength, shift);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda_RightShiftsBlocks launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cuda_RightShiftsBlocks!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result->_mp_d, dev_result, result->_mp_size * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_inputNumber);
	cudaFree(dev_result);
}

void RightShift2(mpz_t result, mpz_t inputNumber, int shift) {

	RightShiftBlocks(result, inputNumber, shift);
}