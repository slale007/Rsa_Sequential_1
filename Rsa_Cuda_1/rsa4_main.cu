/////////////////////////////////////
// My implementation of Montgomery //
/////////////////////////////////////

#include "..\Rsa_Sequential_1\stdafx.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <string.h>

#include "..\Rsa_Cuda_1\rsa1.h"
#include "..\Rsa_Sequential_1\timeutil.h"
#include "..\Rsa_Cuda_1\customFunctions.h"
#include "cudaMpir.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <cuda.h>
#include <device_functions.h>

#define LONGINT int

using namespace std;

clock_t globalTime0;
clock_t globalTime1;
clock_t globalTime2;
clock_t globalTime3;
clock_t globalTime4;
clock_t globalTime5;
clock_t globalTime6;
clock_t globalTime7;
clock_t globalTime8;
clock_t globalTime9;
clock_t start;

// Note: need to test with big strings
char oldMessageForTesting[] = "Stop slacking off.Stop slacking off.Stop slacking off.Stop slacking off.";
char messageForTesting[] = "Nikola Tesla je umro. Umro je siromasan, ali je bio jedan od najkorisnijih ljudi koji su ikada ziveli. Ono sto je stvorio veliko je i, kako vreme prolazi, postaje jos vece";





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
	}*/

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

	for (int k = 1; k < (len+1) / 8; k++) {
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






























































// This one is correct!
void MontgomeryModularMultiplicationV4(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, mpz_t mprim, mpz_t R, int index)
{
	mpz_t t;
	mpz_init(t);
	mpz_t t2;
	mpz_init(t2);
	mpz_t tmp1;
	mpz_init(tmp1);
	mpz_t tmp2;
	mpz_init(tmp2);
	mpz_t tmp3;
	mpz_init(tmp3);
	mpz_t tmp4;
	mpz_init(tmp4);
	mpz_t u;
	mpz_init(u);
	mpz_t slowU;
	mpz_init(slowU);

	// START tmp data
	/*xxx->_mp_d[0] = 256;
	xxx->_mp_size = 1;

	yyy->_mp_d[0] = 256;
	yyy->_mp_size = 1;
	*/
	// END tmp data

	//mpz_mul(t, xxx, yyy);

	MultiplicationInCuda(t, xxx, yyy);

	/*if (mpz_cmp(t, t2) != 0) {
		cout << endl << "Nooooooooooooooooooooooooooooot same" << endl;
	}*/

	mpz_mul(tmp1, t, mprim);

	mpz_tdiv_q_2exp(tmp2, tmp1, index);
	mpz_mul_2exp(tmp2, tmp2, index);
	mpz_sub(tmp2, tmp1, tmp2);

	mpz_mul(tmp3, tmp2, modul);

	mpz_add(tmp4, t, tmp3);

    mpz_tdiv_q_2exp(u, tmp4, index);

	// step 3.
	if (mpz_cmp(u, modul) >= 0)
	{
		mpz_sub(res, u, modul);
	}
	else {
		mpz_add_ui(res, u, 0); // ok
	}
}

void MontgomeryModularExponentiationV4(mpz_t res, mpz_t xxx, mpz_t exponent, mpz_t modul)
{
	mpz_t tempNull;
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);

	// Calculate R and xline = Rmod = R mod modul
	mpz_t RR;
	mpz_t RMod;
	mpz_t Rsquare;
	mpz_t RsquareMod;
	mpz_init(RR);
	mpz_init(RMod);
	mpz_init(Rsquare);
	mpz_init(RsquareMod);
	mpz_add_ui(RR, RR, 1);
	int indexpom0 = 0;

	// Calculate R: R = b ^ messageLength
	mpz_mul_2exp(RR, RR, modul->_mp_size * 64);

	mpz_mul_2exp(Rsquare, RR, 1);
	mpz_mod(RsquareMod, Rsquare, modul);
	mpz_mod(RMod, RR, modul);

	mpz_t mprim;
	mpz_init(mprim);

	mpz_t min1;
	mpz_t onne;
	mpz_init(onne);
	mpz_add_ui(onne, onne, 1);
	mpz_init(min1);
	mpz_sub(min1, tempNull, onne);
	mpz_powm(mprim, modul, min1, RR);

	mpz_sub(mprim, RR, mprim);


	mpz_t xline;
	mpz_init(xline);
	mpz_t xline2pom;
	mpz_init(xline2pom);
	mpz_t xline2;
	mpz_init(xline2);
	mpz_mul(xline2pom, xxx, RR);
	mpz_mod(xline, xline2pom, modul);


	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (exponent->_mp_d[exponent->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int index = 64 * exponent->_mp_size - 64 + indexpom; // ok

	int indexRR = 64 * RR->_mp_size - 64;



	/*MontgomeryModularMultiplicationV4(xline2, xxx, RsquareMod, modul, mprim, RR, indexRR);

	if (mpz_cmp(xline, xline2) != 0) {
		cout << endl << "Nooooooooooooooooooooooooooooot same" << endl;
	}*/

	mpz_mod(res, RR, modul);

	for (int i = index; i >= 0; i--) {
		MontgomeryModularMultiplicationV4(res, res, res, modul, mprim, RR, indexRR);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MontgomeryModularMultiplicationV4(res, res, xline, modul, mprim, RR, indexRR);
		}
	}

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, res);

	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplicationV4(res, AAA, one, modul, mprim, RR, indexRR);
}

void rsaEncryption(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
{
	mpz_t originalMessage, ciphertext, ciphertext2, c_int3;
	mpz_inits(originalMessage, ciphertext, ciphertext2, c_int3, NULL);
	mpz_import(originalMessage,
		messageLength, 
		/* MS word first */ 1,
		/* bytes per word */ 1,
		/* big-endian */ 1,
		/* skip bits */ 0,
		message);

	clock_t startTime = std::clock();
	MontgomeryModularExponentiationV4(
		/* cripted*/ciphertext,
		/* message */ originalMessage,
		/* exponent*/ publicKey->e,
		/* modul*/ publicKey->n);
	cout << "Montgomery realization: "; printTime(startTime);

	startTime = std::clock();
	rsac_encrypt_internal(publicKey, originalMessage, ciphertext2);
	cout << "Mpir realization: "; printTime(startTime);


	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext);
}

void rsaDecryption(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int, m_int2;
	mpz_inits(m_int, c_int, m_int2, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);


	clock_t startTime = std::clock();
	MontgomeryModularExponentiationV4(/*cripted*/m_int,/* message */ c_int, /*exponent*/ priv->d, /*modul*/ priv->n);
	cout << "Montgomery realization: ";
	printTime(startTime);

	// Mpir realization of powm
	startTime = std::clock();
	rsac_decrypt_internal(priv, c_int, m_int);
	cout << "Mpir realization: ";
	printTime(startTime);


	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, m_int2, NULL);
}

void testRsaSequentialMontgomery() {
	char* message = messageForTesting;
	size_t ciphertextLength, messageLength = strlen(message), result_len;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	int fail = 0;
	public_key* publicKey = (public_key*)calloc(sizeof(public_key), 1);
	private_key* privateKey = (private_key*)calloc(sizeof(private_key), 1);

	printf("\n_______________________Key generation_______________________\n\n");

	// Initialize public and private key
	mpz_init(publicKey->n);
	mpz_init(publicKey->e);
	mpz_init(privateKey->n);
	mpz_init(privateKey->e);
	mpz_init(privateKey->d);
	mpz_init(privateKey->p);
	mpz_init(privateKey->q);

	clock_t keygenTime = std::clock();
	rsaKeyGeneration(publicKey, privateKey);
	printTime(keygenTime);

	printf("\n_________________________Encription_________________________\n\n");

	rsaEncryption(publicKey, message, messageLength, c, &ciphertextLength);

	printf("\n_________________________Decription_________________________\n\n");

	rsaDecryption(privateKey, *c, ciphertextLength, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(publicKey);
	free(privateKey);
	free(*c);
	free(*m_result);

	if (fail == 0) {
		printf("\nTest PASSED\n");
	}
}

void printGPUProperties() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		cout << "  Number of multiprocesors: " << prop.multiProcessorCount << endl;
		cout << "  Total global memory: " << prop.totalGlobalMem << endl;
		cout << "  Shared memory per block: " << prop.sharedMemPerBlock << endl;
		cout << "  Number of registers per block per block: " << prop.regsPerBlock << endl;
		cout << "  Warp size: " << prop.warpSize << endl;
		cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
	}
	cout << endl;
}

void printSomeDebuggingStuff() {
	printf("CHAR_BIT je: %d\n", CHAR_BIT);
	printf("Velicina char je: %d\n", sizeof(char));
	printf("Velicina unsigned char je: %d\n", sizeof(unsigned char));
	printf("Velicina unsigned short je: %d\n", sizeof(unsigned short));
	printf("Velicina unsigned int je: %d\n", sizeof(unsigned int));
	printf("Velicina mp_limb_t je: %d\n", sizeof(mp_limb_t));
	printf("Velicina size_t je: %d\n", sizeof(size_t));
	printf("Velicina unsigned long int je: %d\n", sizeof(unsigned long int));
	printf("Velicina unsigned long long je: %d\n", sizeof(unsigned long long));
	printf("Velicina unsigned long long int je: %d\n", sizeof(unsigned long long int));
}

int main() {

	printGPUProperties();
	printSomeDebuggingStuff();

	testRsaSequentialMontgomery();

	return 0;
}