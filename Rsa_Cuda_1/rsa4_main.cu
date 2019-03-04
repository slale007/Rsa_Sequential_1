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
#include "mulSeqBasic.h"
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

clock_t globalTime0 = 0;
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
char messageForTesting[] = "Nikola Tesla je umro. Umro je siromasan, ali je bio jedan od najkorisnijih ljudi koji su ikada ziveli. Ono sto je stvorio veliko je i, kako vreme prolazi, postaje jos vece22................................................................................";
char messageForTesting2048[] = "Nikola Tesla je umro. Umro je siromasan, ali je bio jedan od najkorisnijih ljudi koji su ikada ziveli. Ono sto je stvorio veliko je i, kako vreme prolazi, postaje jos vece22................................................................................Nikola Tesla je umro. Umro je siromasan, ali je bio jedan od najkorisnijih ljudi koji su ikada ziveli. Ono sto je stvorio veliko je i, kako vreme prolazi, postaje jos vece22................................................................................";






#define BIT_SIZE 2048                                   // RSA 2048 
#define RSA_EXP 0x10001                                 // Encryption exponent for RSA

#define NUM_ITER 1                                      // Number of iterations for taking the average time
#define NUM_MSGS (16*1200)                              // Number of messages to submit

#define BLOCK_SIZE 32                                   // BLOCK_SIZE is set to the warp size
#define TOTAL_NUM_THREADS (NUM_MSGS * BLOCK_SIZE)       // TOTAL_NUM_THREADS is the number of total threads to submit

#define WORD_SIZE 32                                    // 32-bit architecture

#define WEXP_WSIZE 5                                    // Windows exponentiation: Window size = 5 bits
#define WEXP_NUM_PRECOMP 32                             // Number of precomputed values for Windows Exponentiation = 2 ^ WEXP_WSIZE

#define MASK32 0x1F                                     // Constant used to wrap around indeces inside the warp
#define LAST_ITEM (BLOCK_SIZE - 1)                      // index that points to the last item inside the warp

#define FREQ_CPU 2200000000                             // Used for measurement with CPU. Not used when GPU timer is used.

#define localidp1 ((threadIdx.x + 1) & MASK32)          // For shifting elements inside the warp, (index + 1) mod 32
#define localidp2 ((threadIdx.x + 2) & MASK32)          // For shifting elements inside the warp, (index + 2) mod 32
#define localidm2 ((threadIdx.x - 2) & MASK32)          // For shifting elements inside the warp, (index - 2) mod 32

typedef unsigned long long int uint64_cu;

// Inline PTX instructions-----------------------------------------------
static inline __device__ uint64_cu addc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("addc.u64 %0, %1, %2;" : "=l" (c) : "l" (a), "l" (b));
	return c;
}

static inline __device__ uint64_cu addc_cc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("addc.cc.u64 %0, %1, %2;" : "=l" (c) : "l" (a), "l" (b));
	return c;
}

static inline __device__ uint64_cu add_cc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("add.cc.u64 %0, %1, %2;" : "=l" (c) : "l" (a), "l" (b));
	return c;
}

#define __mul_hi64(c, x, y) c = __umul64hi(x,y)

// For checking errors----------------------------------------------------
void checkCUDAError(const char *msg)
{
	cudaError_t cuda_error = cudaGetLastError();
	if (cuda_error != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(cuda_error));
		exit(EXIT_FAILURE);
	}
}

// **********************************************************************************
// Modular Multiplication ***********************************************************
// **********************************************************************************

// Long Word Multiplication
// Compute: (regOut1, regOut0) = regB * regA
#define LWMul64(regOut1, regOut0, regB, regA)  { \
    __mul_hi64 (regOut1, regA, regB);					\
    regOut0 = regA * regB;						\
  }

// Lower part of Long Word Multiplication
// Compute: (regOut1, regOut0) = (regB1, regB0) * (regA1, regA0) mod (2^128)
// Temp registers: regTemp2, regTemp1, regTemp0
#define LWHMul64(regOut1, regOut0, regB1, regB0, regA1, regA0, regTemp2, regTemp1, regTemp0) { \
    __mul_hi64 (regTemp2, regA0, regB0);					\
    regTemp1 = regA0 * regB1;						\
    regTemp0 = regA1 * regB0;						\
    regOut0 = regA0 * regB0;						\
    regOut1 = regTemp2 + regTemp1 + regTemp0;				\
  }








// Perform >-two-< Montgomery Multiplication for operands mod p and mod q 
// Inputs and outputs in interleaved representation
// Safe Montgomery Multiplication
__global__ void  MMDouble(
	uint64_cu * local_carries,
	uint64_cu * local_temp,
	const uint64_cu * local_A,
	const uint64_cu * local_B,
	const uint64_cu * local_M,
	uint64_cu * local_MM,
	const uint64_cu reg_M0inv)
{
	uint64_cu i;
	uint64_cu mask;
	uint64_cu qM;
	uint64_cu regTemp1_2;
	uint64_cu regTemp1_1;
	uint64_cu regTemp0;
	uint64_cu reg;
	uint64_cu regC;
	uint64_cu regA = local_A[threadIdx.x];
	uint64_cu regP = local_M[threadIdx.x];
	uint64_cu regB = local_B[0];
	uint64_cu regShared1 = 0;
	uint64_cu regCarries1 = 0;


	local_MM[threadIdx.x] = ~local_M[threadIdx.x];
	if (threadIdx.x == 0) { local_MM[threadIdx.x] |= 1; }  // p is assumed to be odd

	for (i = 0; i < BLOCK_SIZE; i++) {

		LWMul64(regTemp1_1, regTemp0, regA, regB);

		// Accumulate lower part
		regShared1 = add_cc64(regShared1, regTemp0);
		regCarries1 = addc64(regCarries1, 0);

		// Store result to temp vector
		local_temp[threadIdx.x] = regShared1;

		// TODO: this is part of preparation for next iteration - please move it below
		regB = local_B[i + 1];

		// Calculate Montgomery quotient   
		qM = local_temp[0] * reg_M0inv;

		LWMul64(regTemp1_2, regTemp0, qM, regP);
		//////////////////////////// Line of freedom

		// Accumulate lower part 
		local_temp[(threadIdx.x - 1) & MASK32] = add_cc64(regShared1, regTemp0);
		regCarries1 = addc64(regCarries1, 0);

		reg = add_cc64(regTemp1_1, regTemp1_2);
		regC = addc64(0, 0);

		regShared1 = local_temp[threadIdx.x];

		// Carry propagation and reset carry, carry position changes
		regShared1 = add_cc64(regShared1, regCarries1);
		regCarries1 = addc64(0, 0);

		// Accumulate higher part of partial product for Montgomery reduction
		regShared1 = add_cc64(regShared1, reg);
		regCarries1 = addc64(regCarries1, regC);
	}

	// Store results to temp and carry vectors
	local_temp[threadIdx.x] = regShared1;

	if (threadIdx.x < 31) {
		local_carries[threadIdx.x] = regCarries1;
	}
	else {
		local_carries[threadIdx.x] = 0;
	}

	for (i = 15; i > 0; i--) {
		local_temp[(threadIdx.x + 1) & MASK32] = add_cc64(local_temp[(threadIdx.x + 1) & MASK32], local_carries[threadIdx.x]);
		local_carries[(threadIdx.x + 1) & MASK32] = addc64(0, 0);
	}

	if (threadIdx.x > 30) {
		local_carries[threadIdx.x] += regCarries1;
	}

	// If yet carry left, then result is larger than P
	if (local_carries[LAST_ITEM]) {

		mask = -1;

		// Accumulate -P
		local_temp[threadIdx.x] = add_cc64(local_temp[threadIdx.x], local_MM[threadIdx.x] & mask);
		regCarries1 = addc64(local_carries[threadIdx.x], 0);

		if (threadIdx.x < 30) {
			local_carries[threadIdx.x] = regCarries1;
		}
		else {
			local_carries[threadIdx.x] = 0;
		}

		for (i = 15; i > 0; i--) {

			local_temp[((threadIdx.x + 2) & MASK32)] = add_cc64(local_temp[((threadIdx.x + 2) & MASK32)], local_carries[threadIdx.x]);
			local_carries[((threadIdx.x + 1) & MASK32)] = addc64(0, 0);
		}

		if (threadIdx.x > 30) {
			local_carries[threadIdx.x] += (regCarries1 + mask);
		}
	}
}











// **********************************************************************************
// Modular Exponentiation ***********************************************************
// **********************************************************************************

/*


ModExp1 << <NUM_MSGS, BLOCK_SIZE >> > (
	msg_p_d, msg_q_d, // random message mod p
	p_d, q_d, // generated prime p
	p0inv_d, q0inv_d, // this is (-m0)^-1(mod B)
	dp_d, dq_d, // e^-1 mod(p-1)
	rp_d, // 1000...00000 mod(p)// thsi should be increased twice
	r2p_d, // rp_d ^2 mod(p) What about this? Why it is square?
	rq_d, r2q_d,
	result, // probably this is result
	e_p_d, e_q_d
	);


*/

__global__ void __launch_bounds__(32, 8) ModExp1(
	const uint64_cu *message,
	const uint64_cu *modul,
	const uint64_cu *Modul0INV,
	const uint64_cu *RP,
	const uint64_cu *R2P, // R na kvadrat - koristii se samo za prebacivanje u Montgomery formu 
	uint64_cu *global_e_p)// 
{
	// Vectors data
	__shared__ uint64_cu local_temp_vec_p[BLOCK_SIZE];
	__shared__ uint64_cu local_base_p[BLOCK_SIZE];
	__shared__ uint64_cu local_p[BLOCK_SIZE];
	__shared__ uint64_cu local_pp[BLOCK_SIZE];
	__shared__ uint64_cu local_r2p[BLOCK_SIZE];
	// Temp Vectors required for MM
	__shared__ uint64_cu carries_1_p[BLOCK_SIZE];
	__shared__ uint64_cu shared_1_p[BLOCK_SIZE];
	uint64_cu modul0INV, q0inv;

	if ((threadIdx.x & 1) == 0) {
		modul0INV = Modul0INV[2 * blockIdx.x + 1];
		q0inv = Modul0INV[2 * blockIdx.x];
	}
	else {
	}

	int i;

	local_p[threadIdx.x] = modul[blockDim.x * blockIdx.x + threadIdx.x];

	local_base_p[threadIdx.x] = message[blockDim.x * blockIdx.x + threadIdx.x];

	local_r2p[threadIdx.x] = R2P[blockDim.x * blockIdx.x + threadIdx.x];

	// Precompute -p and -q
	local_pp[threadIdx.x] = ~local_p[threadIdx.x];
	if (threadIdx.x == 0) { local_pp[threadIdx.x] |= 1; }  // p is assumed to be odd


	// Transform operand to Montgomery representation------------------------------------------------------
	//MMDouble(
	//	carries_1_p,
	//	shared_1_p,
	//	local_base_p,
	//	local_r2p,
	//	local_p,
	//	local_pp,
	//	modul0INV);

	local_base_p[threadIdx.x] = shared_1_p[threadIdx.x];  // here base_p is correct  

	// Do the precomputations for Windows Exponentiation---------------------------------------------------
	// ^ 0
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x) + threadIdx.x] = RP[blockDim.x * blockIdx.x + threadIdx.x];

	// ^ 1
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 1) + threadIdx.x] = shared_1_p[threadIdx.x];

	// ^ 2
	//MMDouble(carries_1_p, local_temp_vec_p,
	//	local_base_p,
	//	local_base_p,
	//	local_p,
	//	local_pp,
	//	modul0INV);

	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 2) + threadIdx.x] = local_temp_vec_p[threadIdx.x];

	for (i = 3; i < WEXP_NUM_PRECOMP; i++) {

		//MMDouble(carries_1_p, shared_1_p,
		//	local_temp_vec_p,
		//	local_base_p,
		//	local_p,
		//	local_pp,
		//	modul0INV);

		local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];

		global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + i) + threadIdx.x] = shared_1_p[threadIdx.x];
	}

}


























__global__ void cuda_Multiplication(LONGINT* result, unsigned char* first, unsigned char* second, int lengthFirst, int lengthSecond, int* lengthLongResult) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// consider using this again later
	/*__shared__ unsigned char shared_First[18192];
	__shared__ unsigned char shared_Second[18192];

	for (int i = threadIdx.x; i < lengthFirst; i += blockDim.x) {
		shared_First[i] = first[i];
	}

	for (int i = threadIdx.x; i < lengthSecond; i += blockDim.x) {
		shared_Second[i] = second[i];
	}

	__syncthreads();*/

	if (idx < lengthFirst) {
		int m = 0;
		int n = idx;
		int tmp = 0;

		while (n >= 0 && m < lengthSecond) {
			tmp += second[m] * first[n];
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
			tmp += second[m] * first[n];
			m++;
			n--;
		}

		result[idx] = tmp;
	}

	__syncthreads();

	// carry update
	if(idx == 0) {
		int len = *lengthLongResult;
		int carry = 0;
		int tmp = 0;
		int i;
		for (i = 0; i < len; i++) {
			tmp = result[i] + carry;
			carry = tmp >> 8;
			result[i] = tmp & 0xff;
		}

		if (carry != 0) {
			result[i] = carry;
			*lengthLongResult = i + 1;
		}
	}
}

void MultiplicationInCuda(mpz_t result, mpz_t first, mpz_t second) {
	clock_t timeBeforeAndAfter = std::clock();
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
	cudaStatus = cudaMalloc((void**)&dev_first, length1 * sizeof(unsigned char));
	cudaStatus = cudaMalloc((void**)&dev_second, length2 * sizeof(unsigned char));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_first, first->_mp_d, length1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_second, second->_mp_d, length2 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int numberOfThreads = length1 + length2;

	int *midLength;
	int realMidLength = length1 + length2 - 1;


	cudaStatus = cudaMalloc((void**)&midLength, sizeof(int));
	cudaStatus = cudaMemcpy(midLength, &realMidLength, sizeof(int), cudaMemcpyHostToDevice);

	globalTime1 += (clock() - timeBeforeAndAfter);

	// Launch a kernel on the GPU with one thread for each element.
	clock_t timeSpendOnExecutionOnGPU = std::clock();
	cuda_Multiplication <<< 1, numberOfThreads >>>(dev_result, dev_first, dev_second, length1, length2, midLength);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	cudaStatus = cudaDeviceSynchronize();

	globalTime0 += (clock() - timeSpendOnExecutionOnGPU);
	timeBeforeAndAfter = std::clock();

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(tmpRes, dev_result, (length1 + length2) * sizeof(LONGINT), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(&realMidLength, midLength, sizeof(int), cudaMemcpyDeviceToHost);

	result->_mp_d[0] = tmpRes[0];
	result->_mp_d[0] |= tmpRes[1] << 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[2] << 2 * 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[3] << 3 * 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[4] << 4 * 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[5] << 5 * 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[6] << 6 * 8;
	result->_mp_d[0] |= (unsigned long long int)tmpRes[7] << 7 * 8;

	for (int k = 1; k < (realMidLength +1) / 8; k++) {
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

	globalTime1 += (clock() - timeBeforeAndAfter);

Error:
	cudaFree(dev_first);
	cudaFree(dev_second);
	cudaFree(dev_result);
}































// This one is correct!
void MontgomeryModularMultiplicationCUDA(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, mpz_t mprim, mpz_t R, int index)
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

	//mpz_mul(t, xxx, yyy);

	MultiplicationInCuda(t, xxx, yyy);

	mpz_mul(tmp1, t, mprim);

	//MultiplicationInCuda(tmp1, t, mprim);

	/*mpz_tdiv_q_2exp(tmp2, tmp1, index);
	mpz_mul_2exp(tmp2, tmp2, index);
	mpz_sub(tmp2, tmp1, tmp2);*/

	mpz_t indexX;
	mpz_init(indexX);
	mpz_t onne;
	mpz_init(onne);
	mpz_add_ui(onne, onne, 1);

	mpz_mul_2exp(indexX, onne, index);

	mpz_mod(tmp2, tmp1, indexX);

	mpz_mul(tmp3, tmp2, modul);

	mpz_add(tmp4, t, tmp3);

	// Shift right
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

void MontgomeryModularExponentiationCUDA(mpz_t res, mpz_t xxx, mpz_t exponent, mpz_t modul)
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

	mpz_mod(res, RR, modul);

	for (int i = index; i >= 0; i--) {
		MontgomeryModularMultiplicationCUDA(res, res, res, modul, mprim, RR, indexRR);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MontgomeryModularMultiplicationCUDA(res, res, xline, modul, mprim, RR, indexRR);
		}
	}

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, res);

	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplicationCUDA(res, res, one, modul, mprim, RR, indexRR);

	cout << "Time on GPU: " << globalTime0 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Time before and after on  GPU: " << globalTime1 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}
















// This is precomputation
void calcualateBarrettFactor(mpz_t factor, mpz_t xxx, mpz_t modul) {
	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int k = 64 * modul->_mp_size - 64 + indexpom + 1; // ok
	k = k * 2 ;
	mpz_t twoPower;
	mpz_init(twoPower);
	mpz_add_ui(twoPower, twoPower, 1);
	mpz_mul_2exp(twoPower, twoPower, k);

	mpz_div(factor, twoPower, modul);
}


void BarrettExponentiation(mpz_t res, mpz_t xxx, mpz_t exponent, mpz_t modul)
{
	mpz_t tempNull;
	mpz_t factor;
	mpz_init(factor);
	mpz_init(res);
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);
	int indexpom = 0;

	calcualateBarrettFactor(factor, xxx, modul);

	for (int i = 63; i >= 0; i--) {
		if (exponent->_mp_d[exponent->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int index = 64 * exponent->_mp_size - 64 + indexpom; // ok

	for (int i = index; i >= 0; i--) {
		if(res->_mp_size != 0)
		mulSeqBasic(res, res, res);
		//mpz_mul(res, res, res);
		//mpz_mod(res, res, modul);
		BarretModularReductionV2(res, res, modul, factor);
		
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {

			if (res->_mp_size == 0) {
				mpz_add_ui(res, xxx, 0);
			}
			else {
				mulSeqBasic(res, res, xxx);
				//mpz_mul(res, res, xxx);
			}
			BarretModularReductionV2(res, res, modul, factor);
			//mpz_mod(res, res, modul);
		}
	}
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
	mpz_t indexX;
	mpz_init(indexX);
	mpz_t onne;
	mpz_init(onne);

	//mpz_mul(t, xxx, yyy);
	mulSeqBasic(t, xxx, yyy);

	//mpz_mul(tmp1, t, mprim);
	mulSeqBasic(tmp1, t, mprim);


	/*mpz_tdiv_q_2exp(tmp2, tmp1, index);
	mpz_mul_2exp(tmp2, tmp2, index);
	mpz_sub(tmp2, tmp1, tmp2);*/

	mpz_add_ui(onne, onne, 1);
	mpz_mul_2exp(indexX, onne, index);
	mpz_mod(tmp2, tmp1, indexX);

	//mpz_mul(tmp3, tmp2, modul);
	mulSeqBasic(tmp3, tmp2, modul);

	mpz_add(tmp4, t, tmp3);

	// Shift right
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
	mpz_t RR;
	mpz_t BB;
	mpz_t RMod;
	mpz_t Rsquare;
	mpz_t RsquareMod;
	mpz_t mprim;
	mpz_init(mprim);
	mpz_init(RR);
	mpz_init(BB);
	mpz_init(RMod);
	mpz_init(Rsquare);
	mpz_init(RsquareMod);
	mpz_t min1;
	mpz_t onne;
	mpz_init(onne);
	mpz_add_ui(onne, onne, 1);
	mpz_init(min1);
	mpz_sub(min1, tempNull, onne);
	mpz_t xline;
	mpz_init(xline);
	mpz_t xline2pom;
	mpz_init(xline2pom);
	mpz_t xline2;
	mpz_init(xline2);

	// Calculate R and xline = Rmod = R mod modul

	mpz_add_ui(RR, RR, 1);
	mpz_add_ui(BB, BB, 1);

	// Calculate R: R = b ^ messageLength
	mpz_mul_2exp(RR, RR, modul->_mp_size * 64);
	mpz_mul_2exp(BB, BB, modul->_mp_size * 64);
	mpz_mul_2exp(Rsquare, RR, 1);
	mpz_mod(RMod,       RR,      modul);
	mpz_mod(RsquareMod, Rsquare, modul);

	// TODO: Use mpz_invert
	mpz_powm(mprim, modul, min1, BB);
	mpz_sub(mprim, BB, mprim);

	// TODO: This could be separate Montgomery multiplication call
	//mulSeqBasic(xline2pom, xxx, RR);
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
	MontgomeryModularMultiplicationV4(res, res, one, modul, mprim, RR, indexRR);
}
























// __device__ void  MMDouble(
// 	uint64_cu * local_carries,   //done
// 	uint64_cu * local_temp,		 //done
// 	const uint64_cu * local_A,   //done
// 	const uint64_cu * local_B,   //done
// 	const uint64_cu * local_M,   //done
// 	const uint64_cu * local_MM,  //? - local_M
// 	const uint64_cu reg_M0inv)   //?


void MMCuda(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, mpz_t mprim, mpz_t RR, int index) {
	// GPU: Device variables
	uint64_cu * local_carries_GPU;
	uint64_cu * xxx_GPU;
	uint64_cu * yyy_GPU;
	uint64_cu * res_GPU;
	uint64_cu * modul_GPU;
	uint64_cu * minus_modul_GPU;
	uint64_cu * RR_GPU;

	// GPU: Allocate vectors in device memory
	cudaMalloc((void **)&local_carries_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&xxx_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&yyy_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&res_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&modul_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&minus_modul_GPU, sizeof(uint64_cu) * 32);
	cudaMalloc((void **)&RR_GPU, sizeof(uint64_cu) * 32);

	// Copy Host Memory -> Device Memory
	cudaMemcpy(xxx_GPU, xxx, sizeof(uint64_cu) * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(yyy_GPU, yyy, sizeof(uint64_cu) * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(modul_GPU, modul, sizeof(uint64_cu) * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(RR_GPU, RR, sizeof(uint64_cu) * 32, cudaMemcpyHostToDevice);

	MMDouble << < 1, 32 >> >(
		local_carries_GPU,
		res_GPU,
		xxx_GPU,
		yyy_GPU,
		modul_GPU,
		minus_modul_GPU,
		mprim->_mp_d[0]);

	cudaMemcpy(res->_mp_d, res_GPU, sizeof(uint64_cu) * 32, cudaMemcpyDeviceToHost);
}

void MMExpCUDA(mpz_t res, mpz_t xxx, mpz_t exponent, mpz_t modul)
{
	mpz_t tempNull;
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);
	mpz_t RR;
	mpz_t BB;
	mpz_t RMod;
	mpz_t Rsquare;
	mpz_t RsquareMod;
	mpz_t mprim;
	mpz_init(mprim);
	mpz_init(RR);
	mpz_init(BB);
	mpz_init(RMod);
	mpz_init(Rsquare);
	mpz_init(RsquareMod);
	mpz_t min1;
	mpz_t onne;
	mpz_init(onne);
	mpz_add_ui(onne, onne, 1);
	mpz_init(min1);
	mpz_sub(min1, tempNull, onne);
	mpz_t xline;
	mpz_init(xline);
	mpz_t xline2pom;
	mpz_init(xline2pom);
	mpz_t xline2;
	mpz_init(xline2);

	// Calculate R and xline = Rmod = R mod modul

	mpz_add_ui(RR, RR, 1);
	mpz_add_ui(BB, BB, 1);

	// Calculate R: R = b ^ messageLength
	mpz_mul_2exp(RR, RR, modul->_mp_size * 64);
	mpz_mul_2exp(BB, BB, modul->_mp_size * 64);
	mpz_mul_2exp(Rsquare, RR, 1);
	mpz_mod(RMod, RR, modul);
	mpz_mod(RsquareMod, Rsquare, modul);

	// TODO: Use mpz_invert
	mpz_powm(mprim, modul, min1, BB);
	mpz_sub(mprim, BB, mprim);

	// TODO: This could be separate Montgomery multiplication call
	//mulSeqBasic(xline2pom, xxx, RR);
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

	mpz_mod(res, RR, modul);

	mpz_t res_test;
	mpz_init(res_test);
	mpz_mod(res_test, RR, modul);

	for (int i = index; i >= 0; i--) {
		MMCuda(res, res, res, modul, mprim, RR, indexRR);
		MontgomeryModularMultiplicationV4(res_test, res_test, res_test, modul, mprim, RR, indexRR);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MMCuda(res, res, xline, modul, mprim, RR, indexRR);
		}
	}

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, res);

	mpz_add_ui(one, one, 1);
	MMCuda(res, res, one, modul, mprim, RR, indexRR);
}





























// This i probably garbage
// void rsaEncryption(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
// {
// 	mpz_t originalMessage, ciphertext, ciphertext2, c_int3;
// 	mpz_inits(originalMessage, ciphertext, ciphertext2, c_int3, NULL);
// 	mpz_import(originalMessage,
// 		messageLength,
// 		/* MS word first */ 1,
// 		/* bytes per word */ 1,
// 		/* big-endian */ 1,
// 		/* skip bits */ 0,
// 		message);
// 
// 	//clock_t startTime = std::clock();
// 	//MontgomeryModularExponentiationCUDA(
// 	//	/* cripted*/ciphertext,
// 	//	/* message */ originalMessage,
// 	//	/* exponent*/ publicKey->e,
// 	//	/* modul*/ publicKey->n);
// 	///cout << "CUDA Montgomery realization: "; printTime(startTime);
// 
// 	clock_t startTime = std::clock();
// 	BarrettExponentiation(
// 		/* cripted*/ciphertext,
// 		/* message */ originalMessage,
// 		/* exponent*/ publicKey->e,
// 		/* modul*/ publicKey->n);
// 	cout << "Sequential Montgomery realization: "; printTime(startTime);
// 
// 	startTime = std::clock();
// 	MontgomeryModularExponentiationV4(
// 		/* cripted*/ciphertext2,
// 		/* message */ originalMessage,
// 		/* exponent*/ publicKey->e,
// 		/* modul*/ publicKey->n);
// 	cout << "Sequential Montgomery realization: "; printTime(startTime);
// 
// 
// 	startTime = std::clock();
// 	rsac_encrypt_internal(publicKey, originalMessage, ciphertext2);
// 	cout << "Mpir realization: "; printTime(startTime);
// 
// 
// 	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext);
// }
// void rsaDecryption(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
// {
// 	mpz_t m_int, m_int2, m_int3, c_int, c_int1, c_int2;
// 	mpz_inits(m_int, m_int2, m_int3, c_int, c_int1, c_int2, NULL);
// 	mpz_import(
// 		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
// 		/* big-endian */ 1, /* skip bits */ 0, c);
// 	mpz_import(
// 		c_int1, c_len, /* MS word first */ 1, /* bytes per word */ 1,
// 		/* big-endian */ 1, /* skip bits */ 0, c);
// 	mpz_import(
// 		c_int2, c_len, /* MS word first */ 1, /* bytes per word */ 1,
// 		/* big-endian */ 1, /* skip bits */ 0, c);
// 
// 
// 	//clock_t startTime = std::clock();
// 	//MontgomeryModularExponentiationCUDA(/*cripted*/m_int,/* message */ c_int, /*exponent*/ priv->d, /*modul*/ priv->n);
// 	//cout << "CUDA Montgomery realization: ";
// 	//printTime(startTime);
// 
// 	clock_t startTime = std::clock();
// 	BarrettExponentiation(/*cripted*/m_int, /* message */ c_int1, /*exponent*/ priv->d, /*modul*/ priv->n);
// 	cout << "Sequential Barrett realization: ";
// 	printTime(startTime);
// 
// 	startTime = std::clock();
// 	MontgomeryModularExponentiationV4(/*cripted*/m_int2, /* message */ c_int1, /*exponent*/ priv->d, /*modul*/ priv->n);
// 	cout << "Sequential Montgomery realization: ";
// 	printTime(startTime);
// 
// 	// Mpir realization of powm
// 	startTime = std::clock();
// 	rsac_decrypt_internal(priv, c_int2, m_int3);
// 	cout << "Mpir realization: ";
// 	printTime(startTime);
// 
// 
// 	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
// 	mpz_clears(m_int, c_int1, m_int2, NULL);
// }









void debuggingHelperEncription(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
{
	mpz_t originalMessage, ciphertext, ciphertext2, ciphertext3, ciphertext4;
	mpz_inits(originalMessage, ciphertext, ciphertext2, ciphertext3, ciphertext4, NULL);
	mpz_import(originalMessage,
		messageLength,
		/* MS word first */ 1,
		/* bytes per word */ 1,
		/* big-endian */ 1,
		/* skip bits */ 0,
		message);

	clock_t startTime = std::clock();
	BarrettExponentiation(
		/* cripted*/ciphertext,
		/* message */ originalMessage,
		/* exponent*/ publicKey->e,
		/* modul*/ publicKey->n);
	cout << "Barrett realization: "; printTime(startTime);

	startTime = std::clock();
	MontgomeryModularExponentiationV4(
		/* cripted*/ciphertext2,
		/* message */ originalMessage,
		/* exponent*/ publicKey->e,
		/* modul*/ publicKey->n);
	cout << "Sequential Montgomery realization: "; printTime(startTime);


	startTime = std::clock();
	rsac_encrypt_internal(publicKey, originalMessage, ciphertext3);
	cout << "Mpir realization: "; printTime(startTime);

	startTime = std::clock();
	MMExpCUDA(
		/* cripted*/ciphertext4,
		/* message */ originalMessage,
		/* exponent*/ publicKey->e,
		/* modul*/ publicKey->n);
	cout << "Sequential Montgomery realization: "; printTime(startTime);

	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext4);
}





















void rsaEncryptionMontgomerySequential(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
{
	mpz_t originalMessage, ciphertext;
	mpz_inits(originalMessage, ciphertext, NULL);
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
	cout << "Sequential Montgomery realization: "; printTime(startTime);

	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext);
}
void rsaDecryptionMontgomerySequential(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int;
	mpz_inits(m_int, c_int, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);

	clock_t startTime = std::clock();
	MontgomeryModularExponentiationV4(/*cripted*/m_int, /* message */ c_int, /*exponent*/ priv->d, /*modul*/ priv->n);
	cout << "Sequential Montgomery realization: ";
	printTime(startTime);

	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, NULL);
}
void rsaEncryptionMpir(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
{
	mpz_t originalMessage, ciphertext;
	mpz_inits(originalMessage, ciphertext, NULL);
	mpz_import(originalMessage,
		messageLength,
		/* MS word first */ 1,
		/* bytes per word */ 1,
		/* big-endian */ 1,
		/* skip bits */ 0,
		message);

	clock_t startTime = std::clock();
	rsac_encrypt_internal(publicKey, originalMessage, ciphertext);
	cout << "Mpir realization: "; printTime(startTime);

	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext);
}

void rsaDecryptionMpir(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int;
	mpz_inits(m_int, c_int, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);


	// Mpir realization of powm
	clock_t startTime = std::clock();
	rsac_decrypt_internal(priv, c_int, m_int);
	cout << "Mpir realization: ";
	printTime(startTime);


	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, NULL);
}

void rsaEncryptionBarrettSequential(public_key *publicKey, const char *message, size_t messageLength, char **cryptedMessage, size_t *ciphertextLength)
{
	mpz_t originalMessage, ciphertext;
	mpz_inits(originalMessage, ciphertext, NULL);
	mpz_import(originalMessage,
		messageLength, 
		/* MS word first */ 1,
		/* bytes per word */ 1,
		/* big-endian */ 1,
		/* skip bits */ 0,
		message);

	clock_t startTime = std::clock();
	BarrettExponentiation(
		/* cripted*/ciphertext,
		/* message */ originalMessage,
		/* exponent*/ publicKey->e,
		/* modul*/ publicKey->n);
	cout << "Sequential Barrett realization: "; printTime(startTime);

	*cryptedMessage = (char*)mpz_export(NULL, ciphertextLength, 1, 1, 1, 0, ciphertext);
}

void rsaDecryptionBarrettSequential(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int;
	mpz_inits(m_int, c_int, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);

	clock_t startTime = std::clock();
	BarrettExponentiation(/*cripted*/m_int, /* message */ c_int, /*exponent*/ priv->d, /*modul*/ priv->n);
	cout << "Sequential Barrett realization: ";
	printTime(startTime);

	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, NULL);
}










void debugingHelper(public_key* publicKey, private_key* privateKey) {
	char* message = messageForTesting;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	size_t ciphertextLength, messageLength = strlen(message), result_len;

	printf("\n****************************Debugging Helper******************************\n\n");
	printf("\n_________________________Encription_________________________\n\n");

	debuggingHelperEncription(publicKey, message, messageLength, c, &ciphertextLength);

	printf("\n_________________________Decription_________________________\n\n");

	rsaDecryptionMontgomerySequential(privateKey, *c, ciphertextLength, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(*c);
	free(*m_result);
}




void testRsaMontgomerySequential(public_key* publicKey, private_key* privateKey) {
	char* message = messageForTesting;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	size_t ciphertextLength, messageLength = strlen(message), result_len;

	printf("\n****************************Montgomery sequential******************************\n\n");
	printf("\n_________________________Encription_________________________\n\n");

	rsaEncryptionMontgomerySequential(publicKey, message, messageLength, c, &ciphertextLength);

	printf("\n_________________________Decription_________________________\n\n");

	rsaDecryptionMontgomerySequential(privateKey, *c, ciphertextLength, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(*c);
	free(*m_result);
}

void testRsaBarrettSequential(public_key* publicKey, private_key* privateKey) {
	char* message = messageForTesting;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	size_t ciphertextLength, messageLength = strlen(message), result_len;

	printf("\n****************************Barrett sequential******************************\n\n");
	printf("\n_________________________Encription_________________________\n\n");

	rsaEncryptionBarrettSequential(publicKey, message, messageLength, c, &ciphertextLength);

	printf("\n_________________________Decription_________________________\n\n");

	rsaDecryptionBarrettSequential(privateKey, *c, ciphertextLength, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(*c);
	free(*m_result);
}

void testRsaMpir(public_key* publicKey, private_key* privateKey) {
	char* message = messageForTesting;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	size_t ciphertextLength, messageLength = strlen(message), result_len;

	printf("\n****************************Mpir realization******************************\n\n");
	printf("\n_________________________Encription_________________________\n\n");

	rsaEncryptionMpir(publicKey, message, messageLength, c, &ciphertextLength);

	printf("\n_________________________Decription_________________________\n\n");

	rsaDecryptionMpir(privateKey, *c, ciphertextLength, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(*c);
	free(*m_result);
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
		cout << "  Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << endl;
		cout << "  Number of registers per block: " << prop.regsPerBlock << endl;
		cout << "  Warp size: " << prop.warpSize << endl;
		cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
		cout << "  Regs per block: " << prop.regsPerBlock<< endl;
		cout << "  Regs per multiprocessor: " << prop.regsPerMultiprocessor << endl;
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

	debugingHelper(publicKey, privateKey);
	testRsaMontgomerySequential(publicKey, privateKey);
	testRsaMpir(publicKey, privateKey);
	testRsaBarrettSequential(publicKey, privateKey);

	return 0;
}