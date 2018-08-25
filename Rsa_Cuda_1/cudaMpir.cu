#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cudaMpir.h"
#include "device_launch_parameters.h"

// #define LONGINT int

using namespace std;
/*
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
	cuda_RightShiftsBlocks<<<1, inputNumberLength >>>(dev_result, dev_inputNumber, inputNumberLength, shift);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "0cuda_RightShiftsBlocks launch failed: %s\n", cudaGetErrorString(cudaStatus));
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



__global__ void cuda_Multiplication(LONGINT* result, unsigned char* first, unsigned char* second, int lengthFirst, int lengthSecond) {

int idx = blockIdx.x * blockDim.x + threadIdx.x;

__shared__ unsigned char shared_First[2048];
__shared__ unsigned char shared_Second[2048];

for (int i = threadIdx.x; i < lengthFirst; i += blockDim.x) {
shared_First[i] = first[i];
}

for (int i = threadIdx.x; i < lengthSecond; i += blockDim.x) {
shared_Second[i] = second[i];
}

__syncthreads();

if (idx < lengthFirst) {
int m = 0;
int n = idx;
int tmp = 0;

while (n >= 0 && m < lengthSecond) {
tmp += shared_Second[m] * shared_First[n];
m++;
n--;
}

result[idx] = tmp;
}
else if (idx < lengthFirst + lengthSecond - 1) {
int n = lengthFirst - 1;
int m = idx - n;
int tmp = 0;

while (m < lengthSecond && n >= 0) {
tmp += shared_Second[m] * shared_First[n];
m++;
n--;
}

result[idx] = tmp;
}
}

__global__ void cuda_CarryUpdate(LONGINT* longResult, int* lengthLongResult) {

int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx == 0) {
int len = *lengthLongResult;
int carry = 0;
int tmp = 0;
int i;
for (i = 0; i < len; i++) {
tmp = longResult[i] + carry;
carry = tmp >> 8;
longResult[i] = tmp & 0xff;
}

if (carry != 0) {
longResult[i] = carry;
*lengthLongResult = i + 1;
}
}
}

// Is this tested?
//
////
void MultiplicationInCuda(mpz_t result, mpz_t first, mpz_t second) {
cudaError_t cudaStatus;
int length1 = first->_mp_size * 8;
int length2 = second->_mp_size * 8;

mpz_init(result);
result->_mp_size = (length1 + length2);
result->_mp_alloc = result->_mp_size;
result->_mp_d = (unsigned long long int *)malloc(result->_mp_size * sizeof(unsigned long long int));

int *tmpRes = (int *)malloc((length1 + length2) * sizeof(int));

unsigned char* dev_first;
unsigned char* dev_second;
LONGINT* dev_result;

cudaStatus = cudaMalloc((void**)&dev_result, (length1 + length2) * sizeof(LONGINT));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed-1!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_first, length1 * sizeof(unsigned char));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed0!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_second, length2 * sizeof(unsigned char));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed1!");
goto Error;
}

// Copy input vectors from host memory to GPU buffers.
cudaStatus = cudaMemcpy(dev_first, first->_mp_d, length1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy first failed!");
goto Error;
}

cudaStatus = cudaMemcpy(dev_second, second->_mp_d, length2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy second failed!");
goto Error;
}


int numberOfThreads = length1 + length2;

// Launch a kernel on the GPU with one thread for each element.
cuda_Multiplication <<< 1, numberOfThreads >>>(dev_result, dev_first, dev_second, length1, length2);


// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "Multiplication launch failed123: %s\n", cudaGetErrorString(cudaStatus));
goto Error;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "2cudaDeviceSynchronize returned error code %d after launching cuda_RightShiftsBlocks!\n", cudaStatus);
goto Error;
}

// carry update on cuda

/*int *midLength;
int realMidLength = length1 + length2;


cudaStatus = cudaMalloc((void**)&midLength, sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed0!");
goto Error;
}

cudaStatus = cudaMemcpy(midLength, &realMidLength, sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy second failed!");
goto Error;
}

cuda_CarryUpdate <<<1, 1 >>> (dev_result, midLength);

// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "Multiplication launch failed1234: %s\n", cudaGetErrorString(cudaStatus));
goto Error;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "23cudaDeviceSynchronize returned error code %d after launching cuda_RightShiftsBlocks!\n", cudaStatus);
goto Error;
}

// END carry update on cuda 




// Copy output vector from GPU buffer to host memory.
cudaStatus = cudaMemcpy(tmpRes, dev_result, (length1 + length2) * sizeof(LONGINT), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "2cudaMemcpy failed!\n");
	goto Error;
}

int len = length1 + length2 - 1;
unsigned long long int carry = 0;
unsigned long long int tmp = 0;
int i;
for (i = 0; i < len; i++) {
	tmp = tmpRes[i] + carry;
	carry = tmp >> 8;
	tmpRes[i] = tmp & 0xff;
}

if (carry != 0) {
	tmpRes[i] = carry;
	len++;
}

result->_mp_d[0] = tmpRes[0];
result->_mp_d[0] |= tmpRes[1] << 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[2] << 2 * 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[3] << 3 * 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[4] << 4 * 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[5] << 5 * 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[6] << 6 * 8;
result->_mp_d[0] |= (unsigned long long int)tmpRes[7] << 7 * 8;

for (int k = 1; k < (len + 1) / 8; k++) {
	result->_mp_d[k] = 0;
	result->_mp_d[k] |= tmpRes[8 * k];
	result->_mp_d[k] |= tmpRes[8 * k + 1] << 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 2] << 2 * 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 3] << 3 * 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 4] << 4 * 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 5] << 5 * 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 6] << 6 * 8;
	result->_mp_d[k] |= (unsigned long long int)tmpRes[8 * k + 7] << 7 * 8;
}

result->_mp_size = (length1 + length2) / 8;
result->_mp_alloc = (length1 + length2) / 8;

Error:
cudaFree(dev_first);
cudaFree(dev_second);
cudaFree(dev_result);
}



*/