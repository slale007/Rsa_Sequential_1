// GPU code for performing RSA-2048
// Author: Marcelo Kaihara
// Date: 15-03-2011
/*
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

// Inline PTX instructions-----------------------------------------------
static inline __device__ unsigned int addc(unsigned int a, unsigned int b) {
	unsigned int  c;
	asm("addc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ unsigned int addc_cc(unsigned int a, unsigned int b) {
	unsigned int c;
	asm("addc.cc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ unsigned int add_cc(unsigned int a, unsigned int b) {
	unsigned int c;
	asm("add.cc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

#define __add_cc(c, a, b) c = add_cc(a, b);
#define __addc(c, a, b) c = addc(a, b);
#define __mul_hi(c, x, y) c = __umulhi(x,y)

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

// Device code-------------------------------------------------------------
__device__ void Conv_N2I(unsigned int *vectorH, unsigned int *vectorL, unsigned int *local_p, unsigned int *local_q)
// Convert from Normal representation to Interleaved representation
// Inputs: local_p, local_q
// Outputs: vectorH, vectorL
{
	if (threadIdx.x & 1) {
		vectorH[threadIdx.x] = local_q[threadIdx.x];
		vectorL[threadIdx.x] = local_q[threadIdx.x - 1];
	}
	else {
		vectorH[threadIdx.x] = local_p[threadIdx.x + 1];
		vectorL[threadIdx.x] = local_p[threadIdx.x];
	}
}

__device__ void Conv_I2N(unsigned int *local_p, unsigned int *local_q, unsigned int *vectorH, unsigned int *vectorL)
// Convert from Interleaved representation to Normal Representation
// Inputs: vectorH, vectorL
// Outputs: local_p, local_q
{
	if (threadIdx.x & 1) {
		local_q[threadIdx.x] = vectorH[threadIdx.x];
		local_p[threadIdx.x] = vectorH[threadIdx.x - 1];
	}
	else {
		local_q[threadIdx.x] = vectorL[threadIdx.x + 1];
		local_p[threadIdx.x] = vectorL[threadIdx.x];
	}
}

// **********************************************************************************
// Modular Multiplication ***********************************************************
// **********************************************************************************

// Long Word Multiplication
// Compute: (regOut3, regOut2, regOut1, regOut0) = (regB1, regB0) * (regA1, regA0)
// Temp registers: regTemp4, regTemp3, regTemp2, regTemp1, regTemp0
#define LWMul(regOut3, regOut2, regOut1, regOut0, regB1, regB0, regA1, regA0, regTemp4, regTemp3, regTemp2, regTemp1, regTemp0)  { \
    __mul_hi (regTemp3, regA0, regB0);					\
    regTemp1 = regA0 * regB1;						\
    __mul_hi (regTemp2, regA0, regB1);					\
    __mul_hi (regTemp0, regA1, regB0);					\
    __mul_hi (regTemp4, regA1, regB1);					\
    regOut0 = regA0 * regB0;						\
    regOut1 = add_cc (regTemp1, regTemp3);				\
    regOut2 = addc_cc (regTemp0, regTemp2);				\
    regOut3 = addc (regTemp4, 0);					\
    regTemp1 = regA1 * regB0;						\
    regTemp2 = regA1 * regB1;						\
    regOut1 = add_cc (regOut1, regTemp1);				\
    regOut2 = addc_cc (regOut2, regTemp2);				\
    regOut3 = addc (regOut3, 0);					\
  }

// Lower part of Long Word Multiplication
// Compute: (regOut1, regOut0) = (regB1, regB0) * (regA1, regA0) mod (2^64)
// Temp registers: regTemp2, regTemp1, regTemp0
#define LWHMul(regOut1, regOut0, regB1, regB0, regA1, regA0, regTemp2, regTemp1, regTemp0) { \
    __mul_hi (regTemp2, regA0, regB0);					\
    regTemp1 = regA0 * regB1;						\
    regTemp0 = regA1 * regB0;						\
    regOut0 = regA0 * regB0;						\
    regOut1 = regTemp2 + regTemp1 + regTemp0;				\
  }

__device__ void  MMDoubleF(unsigned int * accum_carries, unsigned int * local_carries, unsigned int * local_temp_H, unsigned int * local_temp_L,
	const unsigned int * local_A_H, const unsigned int * local_A_L,
	const unsigned int * local_B_H, const unsigned int * local_B_L,
	const unsigned int * local_M_H, const unsigned int * local_M_L,
	const unsigned int * local_MM_H, const unsigned int * local_MM_L,
	const unsigned int reg_M0inv_H, const unsigned int reg_M0inv_L)
	// Perform two Montgomery Multiplication for operands mod p and mod q 
	// Inputs and outputs in interleaved representation
	// Fast Montgomery Multiplication, returns accumulated remaining carries in accum_carries
{
	int i;

	unsigned int index_is_odd = threadIdx.x & 1;
	unsigned int index_is_odd_p2 = index_is_odd + 2;

	unsigned int qm_A;
	unsigned int qm_B;

	unsigned int mask;

	unsigned int regTemp3_2;
	unsigned int regTemp2_2;
	unsigned int regTemp3_1;
	unsigned int regTemp2_1;
	unsigned int regTemp1;
	unsigned int regTemp0;

	unsigned int regAux5;
	unsigned int regAux4;
	unsigned int regAux3;
	unsigned int regAux2;
	unsigned int regAux1;

	unsigned int regH;
	unsigned int regL;
	unsigned int regC;

	unsigned int regA_A = local_A_H[threadIdx.x];
	unsigned int regA_B = local_A_L[threadIdx.x];

	unsigned int regP_A = local_M_H[threadIdx.x];
	unsigned int regP_B = local_M_L[threadIdx.x];

	unsigned int regB_A = local_B_H[index_is_odd];
	unsigned int regB_B = local_B_L[index_is_odd];

	unsigned int regShared1_A = 0;
	unsigned int regShared1_B = 0;

	unsigned int regCarries1_A = 0;

	for (i = 0; i < BLOCK_SIZE; i += 2) {

		LWMul(regTemp3_2, regTemp2_2, regTemp1, regTemp0, regA_A, regA_B, regB_A, regB_B, regAux1, regAux2, regAux3, regAux4, regAux5);

		// Accumulate lower part
		regShared1_B = add_cc(regShared1_B, regTemp0);
		regShared1_A = addc_cc(regShared1_A, regTemp1);
		regCarries1_A = addc(regCarries1_A, 0);

		// Store result to temp vector
		local_temp_L[threadIdx.x] = regShared1_B;
		local_temp_H[threadIdx.x] = regShared1_A;

		// Calculate Montgomery quotient   
		LWHMul(qm_A, qm_B, local_temp_H[index_is_odd], local_temp_L[index_is_odd], reg_M0inv_H, reg_M0inv_L, regAux1, regAux2, regAux3);

		LWMul(regTemp3_1, regTemp2_1, regTemp1, regTemp0, qm_A, qm_B, regP_A, regP_B, regAux1, regAux2, regAux3, regAux4, regAux5);

		// Accumulate lower part
		local_temp_L[localidm2] = add_cc(regShared1_B, regTemp0);
		local_temp_H[localidm2] = addc_cc(regShared1_A, regTemp1);
		regCarries1_A = addc(regCarries1_A, 0);

		regL = add_cc(regTemp2_1, regTemp2_2);
		regH = addc_cc(regTemp3_1, regTemp3_2);
		regC = addc(0, 0);

		regShared1_B = local_temp_L[threadIdx.x];
		regShared1_A = local_temp_H[threadIdx.x];

		regB_A = local_B_H[i + index_is_odd_p2];
		regB_B = local_B_L[i + index_is_odd_p2];

		// Carry propagation and reset carry, carry position changes
		regShared1_B = add_cc(regShared1_B, regCarries1_A);
		regShared1_A = addc_cc(regShared1_A, 0);
		regCarries1_A = addc(0, 0);

		// Accumulate higher part of partial product for Montgomery reduction
		regShared1_B = add_cc(regShared1_B, regL);
		regShared1_A = addc_cc(regShared1_A, regH);
		regCarries1_A = addc(regCarries1_A, regC);

	}

	// Store results to temp and carry vectors
	local_temp_L[threadIdx.x] = regShared1_B;
	local_temp_H[threadIdx.x] = regShared1_A;

	if (threadIdx.x < 30) {
		local_carries[threadIdx.x] = regCarries1_A;
	}
	else {
		local_carries[threadIdx.x] = 0;
	}

	local_temp_L[localidp2] = add_cc(local_temp_L[localidp2], local_carries[threadIdx.x]);
	local_temp_H[localidp2] = addc_cc(local_temp_H[localidp2], 0);
	local_carries[localidp2] = addc(0, 0);

	accum_carries[threadIdx.x] += local_carries[threadIdx.x];

	if (threadIdx.x > 29) {
		local_carries[threadIdx.x] += regCarries1_A;
	}

	// If yet carry left, then result is larger than P
	if (local_carries[LAST_ITEM] | local_carries[LAST_ITEM - 1]) {

		mask = ((local_carries[LAST_ITEM]) ? (-index_is_odd) : 0) | ((local_carries[LAST_ITEM - 1]) ? (-((threadIdx.x - 1) & 1)) : 0);

		// Accumulate -P
		local_temp_L[threadIdx.x] = add_cc(local_temp_L[threadIdx.x], local_MM_L[threadIdx.x] & mask);
		local_temp_H[threadIdx.x] = addc_cc(local_temp_H[threadIdx.x], local_MM_H[threadIdx.x] & mask);
		regCarries1_A = addc(local_carries[threadIdx.x], 0);

		if (threadIdx.x < 30) {
			local_carries[threadIdx.x] = regCarries1_A;
		}
		else {
			local_carries[threadIdx.x] = 0;
		}

		local_temp_L[localidp2] = add_cc(local_temp_L[localidp2], local_carries[threadIdx.x]);
		local_temp_H[localidp2] = addc_cc(local_temp_H[localidp2], 0);
		local_carries[localidp1] = addc(0, 0);

		accum_carries[threadIdx.x] += local_carries[threadIdx.x];

		if (threadIdx.x > 29) {
			local_carries[threadIdx.x] += (regCarries1_A + mask);
		}
	}
}

__device__ void  MMDouble(unsigned int * local_carries, unsigned int * local_temp_H, unsigned int * local_temp_L,
	const unsigned int * local_A_H, const unsigned int * local_A_L,
	const unsigned int * local_B_H, const unsigned int * local_B_L,
	const unsigned int * local_M_H, const unsigned int * local_M_L,
	const unsigned int * local_MM_H, const unsigned int * local_MM_L,
	const unsigned int reg_M0inv_H, const unsigned int reg_M0inv_L)
	// Perform two Montgomery Multiplication for operands mod p and mod q 
	// Inputs and outputs in interleaved representation
	// Safe Montgomery Multiplication
{

	int i;

	unsigned int mask;

	unsigned int index_is_odd = threadIdx.x & 1;

	unsigned int qm_A;
	unsigned int qm_B;

	unsigned int regTemp3_2;
	unsigned int regTemp2_2;
	unsigned int regTemp3_1;
	unsigned int regTemp2_1;
	unsigned int regTemp1;
	unsigned int regTemp0;

	unsigned int regAux5;
	unsigned int regAux4;
	unsigned int regAux3;
	unsigned int regAux2;
	unsigned int regAux1;

	unsigned int regH;
	unsigned int regL;
	unsigned int regC;

	unsigned int regA_A = local_A_H[threadIdx.x];
	unsigned int regA_B = local_A_L[threadIdx.x];;

	unsigned int regP_A = local_M_H[threadIdx.x];
	unsigned int regP_B = local_M_L[threadIdx.x];

	unsigned int regB_A = local_B_H[index_is_odd];
	unsigned int regB_B = local_B_L[index_is_odd];

	unsigned int regShared1_A = 0;
	unsigned int regShared1_B = 0;

	unsigned int regCarries1_A = 0;

	for (i = 0; i < BLOCK_SIZE; i += 2) {

		LWMul(regTemp3_2, regTemp2_2, regTemp1, regTemp0, regA_A, regA_B, regB_A, regB_B, regAux1, regAux2, regAux3, regAux4, regAux5);

		// Accumulate lower part
		regShared1_B = add_cc(regShared1_B, regTemp0);
		regShared1_A = addc_cc(regShared1_A, regTemp1);
		regCarries1_A = addc(regCarries1_A, 0);

		// Store result to temp vector
		local_temp_L[threadIdx.x] = regShared1_B;
		local_temp_H[threadIdx.x] = regShared1_A;

		regB_A = local_B_H[i + index_is_odd + 2];
		regB_B = local_B_L[i + index_is_odd + 2];

		// Calculate Montgomery quotient   
		LWHMul(qm_A, qm_B, local_temp_H[index_is_odd], local_temp_L[index_is_odd], reg_M0inv_H, reg_M0inv_L, regAux1, regAux2, regAux3);

		LWMul(regTemp3_1, regTemp2_1, regTemp1, regTemp0, qm_A, qm_B, regP_A, regP_B, regAux1, regAux2, regAux3, regAux4, regAux5);

		// Accumulate lower part
		local_temp_L[localidm2] = add_cc(regShared1_B, regTemp0);
		local_temp_H[localidm2] = addc_cc(regShared1_A, regTemp1);
		regCarries1_A = addc(regCarries1_A, 0);

		regL = add_cc(regTemp2_1, regTemp2_2);
		regH = addc_cc(regTemp3_1, regTemp3_2);
		regC = addc(0, 0);

		regShared1_B = local_temp_L[threadIdx.x];
		regShared1_A = local_temp_H[threadIdx.x];

		// Carry propagation and reset carry, carry position changes
		regShared1_B = add_cc(regShared1_B, regCarries1_A);
		regShared1_A = addc_cc(regShared1_A, 0);
		regCarries1_A = addc(0, 0);

		// Accumulate higher part of partial product for Montgomery reduction
		regShared1_B = add_cc(regShared1_B, regL);
		regShared1_A = addc_cc(regShared1_A, regH);
		regCarries1_A = addc(regCarries1_A, regC);

	}

	// Store results to temp and carry vectors
	local_temp_L[threadIdx.x] = regShared1_B;
	local_temp_H[threadIdx.x] = regShared1_A;

	if (threadIdx.x < 30) {
		local_carries[threadIdx.x] = regCarries1_A;
	}
	else {
		local_carries[threadIdx.x] = 0;
	}

	for (i = 15; i > 0; i--) {

		local_temp_L[localidp2] = add_cc(local_temp_L[localidp2], local_carries[threadIdx.x]);
		local_temp_H[localidp2] = addc_cc(local_temp_H[localidp2], 0);
		local_carries[localidp2] = addc(0, 0);

	}

	if (threadIdx.x > 29) {
		local_carries[threadIdx.x] += regCarries1_A;
	}

	// If yet carry left, then result is larger than P
	if (local_carries[LAST_ITEM] | local_carries[LAST_ITEM - 1]) {

		mask = ((local_carries[LAST_ITEM]) ? (-index_is_odd) : 0) | ((local_carries[LAST_ITEM - 1]) ? (-((threadIdx.x - 1) & 1)) : 0);

		// Accumulate -P
		local_temp_L[threadIdx.x] = add_cc(local_temp_L[threadIdx.x], local_MM_L[threadIdx.x] & mask);
		local_temp_H[threadIdx.x] = addc_cc(local_temp_H[threadIdx.x], local_MM_H[threadIdx.x] & mask);
		regCarries1_A = addc(local_carries[threadIdx.x], 0);

		if (threadIdx.x < 30) {
			local_carries[threadIdx.x] = regCarries1_A;
		}
		else {
			local_carries[threadIdx.x] = 0;
		}

		for (i = 15; i > 0; i--) {

			local_temp_L[localidp2] = add_cc(local_temp_L[localidp2], local_carries[threadIdx.x]);
			local_temp_H[localidp2] = addc_cc(local_temp_H[localidp2], 0);
			local_carries[localidp1] = addc(0, 0);
		}

		if (threadIdx.x > 29) {
			local_carries[threadIdx.x] += (regCarries1_A + mask);
		}
	}
}

// **********************************************************************************
// Modular Exponentiation ***********************************************************
// **********************************************************************************

#define __exp_step(index1, index2) {					\
    local_e_p [threadIdx.x] = global_e_p [blockDim.x * ( WEXP_NUM_PRECOMP * blockIdx.x + (index1)) + threadIdx.x]; \
    local_e_q [threadIdx.x] = global_e_q [blockDim.x * ( WEXP_NUM_PRECOMP * blockIdx.x + (index2)) + threadIdx.x]; \
    Conv_N2I (tempA, tempB, local_e_p, local_e_q);			\
    for (j = WEXP_WSIZE - 1; j >= 0; j--) {				\
      MMDoubleF (local_temp_carry_q, carries_1_p, shared_1_p, shared_1_q, local_temp_vec_p, local_temp_vec_q, local_temp_vec_p, local_temp_vec_q, local_p, local_q, local_pp, local_qq, p0inv, q0inv); \
      local_temp_vec_p [threadIdx.x] = shared_1_p [threadIdx.x];	\
      local_temp_vec_q [threadIdx.x] = shared_1_q [threadIdx.x];	\
    }									\
    MMDoubleF (local_temp_carry_q, carries_1_p, shared_1_p, shared_1_q, local_temp_vec_p, local_temp_vec_q, tempA, tempB, local_p, local_q, local_pp, local_qq, p0inv, q0inv); \
    local_temp_vec_p [threadIdx.x] = shared_1_p [threadIdx.x];		\
    local_temp_vec_q [threadIdx.x] = shared_1_q [threadIdx.x];		\
  }

__global__ void __launch_bounds__(32, 8) ModExp1(const unsigned int *A_p, const unsigned int *A_q,
	const unsigned int *P, const unsigned int *Q,
	const unsigned int *P0INV, const unsigned int *Q0INV,
	const unsigned int *DMP1, const unsigned int *DMQ1,
	const unsigned int *RP, const unsigned int *R2P,
	const unsigned int *RQ, const unsigned int *R2Q,
	unsigned int *Z_p, unsigned int *Z_q,
	unsigned int *global_e_p, unsigned int *global_e_q)
{

	// Vectors data
	__shared__ unsigned int local_temp_vec_p[BLOCK_SIZE];
	__shared__ unsigned int local_temp_vec_q[BLOCK_SIZE];

	__shared__ unsigned int local_base_p[BLOCK_SIZE];
	__shared__ unsigned int local_base_q[BLOCK_SIZE];

	__shared__ unsigned int local_p[BLOCK_SIZE];
	__shared__ unsigned int local_pp[BLOCK_SIZE];

	__shared__ unsigned int local_q[BLOCK_SIZE];
	__shared__ unsigned int local_qq[BLOCK_SIZE];

	__shared__ unsigned int local_r2p[BLOCK_SIZE];
	__shared__ unsigned int local_r2q[BLOCK_SIZE];

	// Temp Vectors required for MM
	__shared__ unsigned int carries_1_p[BLOCK_SIZE];
	__shared__ unsigned int shared_1_p[BLOCK_SIZE];

	__shared__ unsigned int shared_1_q[BLOCK_SIZE];
	__shared__ unsigned int local_temp_carry_q[BLOCK_SIZE];

	// Vectors for precomputatin (Windows Exp)
	__shared__ unsigned int tempA[BLOCK_SIZE];
	__shared__ unsigned int tempB[BLOCK_SIZE];

	unsigned int p0inv, q0inv;

	if ((threadIdx.x & 1) == 0) {
		p0inv = P0INV[2 * blockIdx.x + 1];
		q0inv = P0INV[2 * blockIdx.x];
	}
	else {
		p0inv = Q0INV[2 * blockIdx.x + 1];
		q0inv = Q0INV[2 * blockIdx.x];
	}

	int i;

	local_p[threadIdx.x] = P[blockDim.x * blockIdx.x + threadIdx.x];
	local_q[threadIdx.x] = Q[blockDim.x * blockIdx.x + threadIdx.x];

	local_base_p[threadIdx.x] = A_p[blockDim.x * blockIdx.x + threadIdx.x];
	local_base_q[threadIdx.x] = A_q[blockDim.x * blockIdx.x + threadIdx.x];

	local_r2p[threadIdx.x] = R2P[blockDim.x * blockIdx.x + threadIdx.x];
	local_r2q[threadIdx.x] = R2Q[blockDim.x * blockIdx.x + threadIdx.x];

	// Precompute -p and -q
	local_pp[threadIdx.x] = ~local_p[threadIdx.x];
	if (threadIdx.x == 0) { local_pp[threadIdx.x] |= 1; }  // p is assumed to be odd
	local_qq[threadIdx.x] = ~local_q[threadIdx.x];
	if (threadIdx.x == 0) { local_qq[threadIdx.x] |= 1; }  // q is assumed to be odd


														   // Convert to Folded representation ---------------------------------------------------------------
	Conv_N2I(tempA, tempB, local_base_p, local_base_q);

	local_base_p[threadIdx.x] = tempA[threadIdx.x];
	local_base_q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_r2p, local_r2q);

	local_r2p[threadIdx.x] = tempA[threadIdx.x];
	local_r2q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_p, local_q);

	local_p[threadIdx.x] = tempA[threadIdx.x];
	local_q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_pp, local_qq);

	local_pp[threadIdx.x] = tempA[threadIdx.x];
	local_qq[threadIdx.x] = tempB[threadIdx.x];

	// Transform operand to Montgomery representation------------------------------------------------------
	MMDouble(carries_1_p, shared_1_p, shared_1_q,
		local_base_p, local_base_q,
		local_r2p, local_r2q,
		local_p, local_q,
		local_pp, local_qq,
		p0inv, q0inv);

	local_base_p[threadIdx.x] = shared_1_p[threadIdx.x];  // here base_p is correct  
	local_base_q[threadIdx.x] = shared_1_q[threadIdx.x];  // here base_q is correct

	Conv_I2N(tempA, tempB, shared_1_p, shared_1_q);

	// Do the precomputations for Windows Exponentiation---------------------------------------------------
	// ^ 0
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x) + threadIdx.x] = RP[blockDim.x * blockIdx.x + threadIdx.x];
	global_e_q[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x) + threadIdx.x] = RQ[blockDim.x * blockIdx.x + threadIdx.x];

	// ^ 1
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 1) + threadIdx.x] = tempA[threadIdx.x];
	global_e_q[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 1) + threadIdx.x] = tempB[threadIdx.x];

	// ^ 2
	MMDoubleF(local_temp_carry_q, carries_1_p, local_temp_vec_p, local_temp_vec_q,
		local_base_p, local_base_q,
		local_base_p, local_base_q,
		local_p, local_q,
		local_pp, local_qq,
		p0inv, q0inv);

	Conv_I2N(tempA, tempB, local_temp_vec_p, local_temp_vec_q);

	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 2) + threadIdx.x] = tempA[threadIdx.x];
	global_e_q[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 2) + threadIdx.x] = tempB[threadIdx.x];

	for (i = 3; i < WEXP_NUM_PRECOMP; i++) {

		MMDoubleF(local_temp_carry_q, carries_1_p, shared_1_p, shared_1_q,
			local_temp_vec_p, local_temp_vec_q,
			local_base_p, local_base_q,
			local_p, local_q,
			local_pp, local_qq,
			p0inv, q0inv);

		Conv_I2N(tempA, tempB, shared_1_p, shared_1_q);

		local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];
		local_temp_vec_q[threadIdx.x] = shared_1_q[threadIdx.x];

		global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + i) + threadIdx.x] = tempA[threadIdx.x];
		global_e_q[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + i) + threadIdx.x] = tempB[threadIdx.x];

	}

}


__global__ void __launch_bounds__(32, 8) ModExp2(const unsigned int *A_p,
	const unsigned int *A_q,

	const unsigned int *P,
	const unsigned int *Q,

	const unsigned int *P0INV,
	const unsigned int *Q0INV,

	const unsigned int *DMP1,
	const unsigned int *DMQ1,

	const unsigned int *RP,
	const unsigned int *R2P,

	const unsigned int *RQ,
	const unsigned int *R2Q,

	unsigned int *Z_p,
	unsigned int *Z_q,

	unsigned int *global_e_p,
	unsigned int *global_e_q)
{

	// Vectors data
	__shared__ unsigned int local_temp_vec_p[BLOCK_SIZE];
	__shared__ unsigned int local_temp_vec_q[BLOCK_SIZE];

	__shared__ unsigned int local_base_p[BLOCK_SIZE];
	__shared__ unsigned int local_base_q[BLOCK_SIZE];

	__shared__ unsigned int local_p[BLOCK_SIZE];
	__shared__ unsigned int local_pp[BLOCK_SIZE];

	__shared__ unsigned int local_q[BLOCK_SIZE];
	__shared__ unsigned int local_qq[BLOCK_SIZE];

	__shared__ unsigned int local_dmp1[BLOCK_SIZE];
	__shared__ unsigned int local_dmq1[BLOCK_SIZE];

	__shared__ unsigned int local_r2p[BLOCK_SIZE];
	__shared__ unsigned int local_r2q[BLOCK_SIZE];

	// Temp Vectors required for MM
	__shared__ unsigned int carries_1_p[BLOCK_SIZE];
	__shared__ unsigned int shared_1_p[BLOCK_SIZE];

	__shared__ unsigned int shared_1_q[BLOCK_SIZE];
	__shared__ unsigned int local_temp_carry_q[BLOCK_SIZE];

	// Vectors for precomputatin (Windows Exp)
	__shared__ unsigned int local_e_p[BLOCK_SIZE];
	__shared__ unsigned int local_e_q[BLOCK_SIZE];

	__shared__ unsigned int tempA[BLOCK_SIZE];
	__shared__ unsigned int tempB[BLOCK_SIZE];

	unsigned int index1p, index1q;
	unsigned int index2p, index2q;

	unsigned int p0inv, q0inv;

	if ((threadIdx.x & 1) == 0) {
		p0inv = P0INV[2 * blockIdx.x + 1];
		q0inv = P0INV[2 * blockIdx.x];
	}
	else {
		p0inv = Q0INV[2 * blockIdx.x + 1];
		q0inv = Q0INV[2 * blockIdx.x];
	}

	int i;
	int j;

	local_p[threadIdx.x] = P[blockDim.x * blockIdx.x + threadIdx.x];
	local_q[threadIdx.x] = Q[blockDim.x * blockIdx.x + threadIdx.x];

	local_base_p[threadIdx.x] = A_p[blockDim.x * blockIdx.x + threadIdx.x];
	local_base_q[threadIdx.x] = A_q[blockDim.x * blockIdx.x + threadIdx.x];

	local_r2p[threadIdx.x] = R2P[blockDim.x * blockIdx.x + threadIdx.x];
	local_r2q[threadIdx.x] = R2Q[blockDim.x * blockIdx.x + threadIdx.x];

	local_dmp1[threadIdx.x] = DMP1[blockDim.x * blockIdx.x + threadIdx.x];
	local_dmq1[threadIdx.x] = DMQ1[blockDim.x * blockIdx.x + threadIdx.x];

	// Precompute -p and -q
	local_pp[threadIdx.x] = ~local_p[threadIdx.x];
	if (threadIdx.x == 0) { local_pp[threadIdx.x] |= 1; }  // p is assumed to be odd
	local_qq[threadIdx.x] = ~local_q[threadIdx.x];
	if (threadIdx.x == 0) { local_qq[threadIdx.x] |= 1; }  // q is assumed to be odd


														   // Convert to Folded representation ---------------------------------------------------------------
	Conv_N2I(tempA, tempB, local_base_p, local_base_q);

	local_base_p[threadIdx.x] = tempA[threadIdx.x];
	local_base_q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_r2p, local_r2q);

	local_r2p[threadIdx.x] = tempA[threadIdx.x];
	local_r2q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_p, local_q);

	local_p[threadIdx.x] = tempA[threadIdx.x];
	local_q[threadIdx.x] = tempB[threadIdx.x];

	Conv_N2I(tempA, tempB, local_pp, local_qq);

	local_pp[threadIdx.x] = tempA[threadIdx.x];
	local_qq[threadIdx.x] = tempB[threadIdx.x];

	// Initialize vectors----------------------------------------------------------------------
	carries_1_p[threadIdx.x] = 0;
	shared_1_p[threadIdx.x] = 0;

	shared_1_q[threadIdx.x] = 0;
	local_temp_carry_q[threadIdx.x] = 0;

	// Load data from global memory
	local_temp_vec_p[threadIdx.x] = RP[blockDim.x * blockIdx.x + threadIdx.x];
	local_temp_vec_q[threadIdx.x] = RQ[blockDim.x * blockIdx.x + threadIdx.x];

	Conv_N2I(tempA, tempB, local_temp_vec_p, local_temp_vec_q);

	local_temp_vec_p[threadIdx.x] = tempA[threadIdx.x];  // here it contains RP
	local_temp_vec_q[threadIdx.x] = tempB[threadIdx.x];  // here it contains RQ

														 //------------------------------------------------------------------------------------------
	for (i = LAST_ITEM; i >= 6; i -= 5) {

		index1p = local_dmp1[i]; index1q = local_dmq1[i];

		__exp_step(index1p >> 27, index1q >> 27);

		index2p = local_dmp1[i - 1]; index2q = local_dmq1[i - 1];

		__exp_step((index1p >> 22) & 0x1F, (index1q >> 22) & 0x1F);
		__exp_step((index1p >> 17) & 0x1F, (index1q >> 17) & 0x1F);
		__exp_step((index1p >> 12) & 0x1F, (index1q >> 12) & 0x1F);
		__exp_step((index1p >> 7) & 0x1F, (index1q >> 7) & 0x1F);
		__exp_step((index1p >> 2) & 0x1F, (index1q >> 2) & 0x1F);

		__exp_step(((index1p << 3) | (index2p >> 29)) & 0x1F, ((index1q << 3) | (index2q >> 29)) & 0x1F);
		__exp_step((index2p >> 24) & 0x1F, (index2q >> 24) & 0x1F);
		__exp_step((index2p >> 19) & 0x1F, (index2q >> 19) & 0x1F);
		__exp_step((index2p >> 14) & 0x1F, (index2q >> 14) & 0x1F);
		__exp_step((index2p >> 9) & 0x1F, (index2q >> 9) & 0x1F);

		index1p = local_dmp1[i - 2]; index1q = local_dmq1[i - 2];

		__exp_step((index2p >> 4) & 0x1F, (index2q >> 4) & 0x1F);

		__exp_step(((index2p << 1) | (index1p >> 31)) & 0x1F, ((index2q << 1) | (index1q >> 31)) & 0x1F);
		__exp_step((index1p >> 26) & 0x1F, (index1q >> 26) & 0x1F);
		__exp_step((index1p >> 21) & 0x1F, (index1q >> 21) & 0x1F);
		__exp_step((index1p >> 16) & 0x1F, (index1q >> 16) & 0x1F);
		__exp_step((index1p >> 11) & 0x1F, (index1q >> 11) & 0x1F);
		__exp_step((index1p >> 6) & 0x1F, (index1q >> 6) & 0x1F);

		index2p = local_dmp1[i - 3]; index2q = local_dmq1[i - 3];

		__exp_step((index1p >> 1) & 0x1F, (index1q >> 1) & 0x1F);

		__exp_step(((index1p << 4) | (index2p >> 28)) & 0x1F, ((index1q << 4) | (index2q >> 28)) & 0x1F);
		__exp_step((index2p >> 23) & 0x1F, (index2q >> 23) & 0x1F);
		__exp_step((index2p >> 18) & 0x1F, (index2q >> 18) & 0x1F);
		__exp_step((index2p >> 13) & 0x1F, (index2q >> 13) & 0x1F);
		__exp_step((index2p >> 8) & 0x1F, (index2q >> 8) & 0x1F);

		index1p = local_dmp1[i - 4]; index1q = local_dmq1[i - 4];

		__exp_step((index2p >> 3) & 0x1F, (index2q >> 3) & 0x1F);

		__exp_step(((index2p << 2) | (index1p >> 30)) & 0x1F, ((index2q << 2) | (index1q >> 30)) & 0x1F);
		__exp_step((index1p >> 25) & 0x1F, (index1q >> 25) & 0x1F);
		__exp_step((index1p >> 20) & 0x1F, (index1q >> 20) & 0x1F);
		__exp_step((index1p >> 15) & 0x1F, (index1q >> 15) & 0x1F);
		__exp_step((index1p >> 10) & 0x1F, (index1q >> 10) & 0x1F);
		__exp_step((index1p >> 5) & 0x1F, (index1q >> 5) & 0x1F);
		__exp_step((index1p) & 0x1F, (index1q) & 0x1F);

	}

	index1p = local_dmp1[i]; index1q = local_dmq1[i];

	__exp_step(index1p >> 27, index1q >> 27);
	__exp_step((index1p >> 22) & 0x1F, (index1q >> 22) & 0x1F);
	__exp_step((index1p >> 17) & 0x1F, (index1q >> 17) & 0x1F);
	__exp_step((index1p >> 12) & 0x1F, (index1q >> 12) & 0x1F);
	__exp_step((index1p >> 7) & 0x1F, (index1q >> 7) & 0x1F);

	index2p = local_dmp1[i - 1]; index2q = local_dmq1[i - 1];

	__exp_step((index1p >> 2) & 0x1F, (index1q >> 2) & 0x1F);

	__exp_step(((index1p << 3) | (index2p >> 29)) & 0x1F, ((index1q << 3) | (index2q >> 29)) & 0x1F);
	__exp_step((index2p >> 24) & 0x1F, (index2q >> 24) & 0x1F);
	__exp_step((index2p >> 19) & 0x1F, (index2q >> 19) & 0x1F);
	__exp_step((index2p >> 14) & 0x1F, (index2q >> 14) & 0x1F);
	__exp_step((index2p >> 9) & 0x1F, (index2q >> 9) & 0x1F);
	__exp_step((index2p >> 4) & 0x1F, (index2q >> 4) & 0x1F);

	local_e_p[threadIdx.x] = global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + (index2p & 0xF)) + threadIdx.x]; // load precomputed value in folded representation
	local_e_q[threadIdx.x] = global_e_q[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + (index2q & 0xF)) + threadIdx.x]; // load precomputed value in folded representation

	Conv_N2I(tempA, tempB, local_e_p, local_e_q);

	for(j = WEXP_WSIZE - 2; j >= 0; j--) {
		MMDoubleF(local_temp_carry_q, carries_1_p, shared_1_p, shared_1_q,
			local_temp_vec_p, local_temp_vec_q,
			local_temp_vec_p, local_temp_vec_q,
			local_p, local_q,
			local_pp, local_qq,
			p0inv, q0inv);
		local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];
		local_temp_vec_q[threadIdx.x] = shared_1_q[threadIdx.x];
	}
	MMDoubleF(local_temp_carry_q, carries_1_p, shared_1_p, shared_1_q,
		local_temp_vec_p, local_temp_vec_q,
		tempA, tempB,
		local_p, local_q,
		local_pp, local_qq,
		p0inv, q0inv);

	local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];
	local_temp_vec_q[threadIdx.x] = shared_1_q[threadIdx.x];

	// Transform back to original representation
	if (threadIdx.x < 2) {
		local_r2p[threadIdx.x] = 0;
		local_r2q[threadIdx.x] = 1;
	}
	else {
		local_r2p[threadIdx.x] = 0;
		local_r2q[threadIdx.x] = 0;
	}

	MMDouble(carries_1_p, shared_1_p, shared_1_q,
		local_temp_vec_p, local_temp_vec_q,
		local_r2p, local_r2q,
		local_p, local_q,
		local_pp, local_qq,
		p0inv, q0inv);

	// Convert backwards from Folded representation------------------------------------------------------
	Conv_I2N(tempA, tempB, shared_1_p, shared_1_q);

	shared_1_p[threadIdx.x] = tempA[threadIdx.x];
	shared_1_q[threadIdx.x] = tempB[threadIdx.x];

	// Store result in global memory
	Z_p[blockDim.x * blockIdx.x + threadIdx.x] = shared_1_p[threadIdx.x];
	Z_q[blockDim.x * blockIdx.x + threadIdx.x] = shared_1_q[threadIdx.x];

}

void fprint_mpz(mpz_t x) {

	int i;
	unsigned long int aux;
	for (i = 15; i >= 0; i--) {
		aux = mpz_getlimbn(x, i);
		printf("%8x ", aux >> 32);
		printf("%8x ", aux & 0xFFFFFFFF);
	}
	printf("\n");
}

// Host code-------------------------------------------------------------
int mainx(int argc, char** argv)
{

	printf("Benchmark: RSA2048 on GPU (Optimized for Fermi)\n");

	int i, j;

	float timeExp = 0;
	float delay = 0;
	float accum_delay = 0;
	float throughput = 0;

	cudaEvent_t start, stop;

	// GMP variables
	mpz_t * msg_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);

	mpz_t * msg_p_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * msg_q_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);

	mpz_t * smsg_gmp_p = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * smsg_gmp_gpu_p = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);

	mpz_t * smsg_gmp_q = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * smsg_gmp_gpu_q = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);

	mpz_t * smsg_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * smsg_gmp_gpu = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);


	for (i = 0; i < NUM_MSGS; i++) {
		mpz_init(msg_gmp[i]);

		mpz_init(msg_p_gmp[i]);
		mpz_init(msg_q_gmp[i]);

		mpz_init(smsg_gmp_p[i]);
		mpz_init(smsg_gmp_gpu_p[i]);

		mpz_init(smsg_gmp_q[i]);
		mpz_init(smsg_gmp_gpu_q[i]);

		mpz_init(smsg_gmp[i]);
		mpz_init(smsg_gmp_gpu[i]);
	}

	mpz_t temp_gmp;
	mpz_init(temp_gmp);

	gmp_randstate_t rnd_state;
	gmp_randinit_default(rnd_state);
	gmp_randseed_ui(rnd_state, 598182);


	// Host variables
	unsigned int * msg_p_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * msg_q_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);

	unsigned int * dp_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * dq_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);

	unsigned int * p_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * q_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);

	unsigned int * rp_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * r2p_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * p0inv_h = (unsigned int *)malloc(sizeof(unsigned int) * 2 * NUM_MSGS);

	unsigned int * rq_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * r2q_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * q0inv_h = (unsigned int *)malloc(sizeof(unsigned int) * 2 * NUM_MSGS);

	unsigned int * q_i_p_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);

	unsigned int * smsg_p_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);
	unsigned int * smsg_q_h = (unsigned int *)malloc(sizeof(unsigned int) * TOTAL_NUM_THREADS);


	// Device variables
	unsigned int * msg_p_d;
	unsigned int * msg_q_d;

	unsigned int * dp_d;
	unsigned int * dq_d;

	unsigned int * p_d;
	unsigned int * q_d;

	unsigned int * p0inv_d;
	unsigned int * q0inv_d;

	unsigned int * rp_d;
	unsigned int * r2p_d;

	unsigned int * rq_d;
	unsigned int * r2q_d;

	unsigned int * smsg_p_d;
	unsigned int * smsg_q_d;

	unsigned int * e_p_d;
	unsigned int * e_q_d;

	unsigned int * q_i_p_d;


	printf("Initializing device\n"); fflush(stdout);

	// Allocate vectors in device memory
	cudaMalloc((void **)&msg_p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&msg_q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&dp_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&dq_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&p0inv_d, sizeof(unsigned int) * 2 * NUM_MSGS);
	cudaMalloc((void **)&q0inv_d, sizeof(unsigned int) * 2 * NUM_MSGS);

	cudaMalloc((void **)&rp_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&r2p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&rq_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&r2q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&smsg_p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&smsg_q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	cudaMalloc((void **)&e_p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS * WEXP_NUM_PRECOMP);
	cudaMalloc((void **)&e_q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS * WEXP_NUM_PRECOMP);

	cudaMalloc((void **)&q_i_p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS);

	// Generate parameters
	mpz_t p, q, pq, pm1, qm1, phi, e, d, p_i_q, q_i_p, dp, dq;
	mpz_t p0inv, q0inv;
	mpz_t pow2_ws;
	mpz_t rp, rq, r2p, r2q;

	mpz_init(p);
	mpz_init(q);
	mpz_init(pq);

	do
	{
		mpz_urandomb(p, rnd_state, BIT_SIZE / 2);
		mpz_setbit(p, BIT_SIZE / 2 - 1);
		mpz_setbit(p, BIT_SIZE / 2 - 2);
		mpz_setbit(p, 0);

		mpz_urandomb(q, rnd_state, BIT_SIZE / 2);
		mpz_setbit(q, BIT_SIZE / 2 - 1);
		mpz_setbit(q, BIT_SIZE / 2 - 2);
		mpz_setbit(q, 0);

		mpz_gcd(pq, p, q);
	} while (mpz_cmp_ui(pq, 1) != 0);

	mpz_mul(pq, p, q);

	mpz_init_set_ui(e, RSA_EXP);
	mpz_init(d);
	mpz_init(pm1);
	mpz_init(qm1);
	mpz_init(phi);

	mpz_sub_ui(pm1, p, 1);
	mpz_sub_ui(qm1, q, 1);
	mpz_mul(phi, pm1, qm1);
	if (mpz_invert(d, e, phi) == 0)
		abort();

	mpz_init(p_i_q);
	if (mpz_invert(p_i_q, p, q) == 0)
		abort();

	mpz_init(q_i_p);
	if (mpz_invert(q_i_p, q, p) == 0)
		abort();

	mpz_init(dp);
	mpz_init(dq);
	mpz_mod(dp, d, pm1);
	mpz_mod(dq, d, qm1);

	mpz_init(p0inv);
	mpz_init(q0inv);
	mpz_init(pow2_ws);
	mpz_setbit(pow2_ws, 2 * WORD_SIZE);
	mpz_neg(temp_gmp, p);
	mpz_invert(p0inv, temp_gmp, pow2_ws);
	mpz_neg(temp_gmp, q);
	mpz_invert(q0inv, temp_gmp, pow2_ws);

	mpz_init(rp);
	mpz_init(rq);
	mpz_set_ui(rp, 0);
	mpz_setbit(rp, 1024);
	mpz_set(rq, rp);
	mpz_mod(rp, rp, p);
	mpz_mod(rq, rq, q);
	mpz_init(r2p);
	mpz_init(r2q);
	mpz_mul(r2p, rp, rp);
	mpz_mod(r2p, r2p, p);
	mpz_mul(r2q, rq, rq);
	mpz_mod(r2q, r2q, q);

	// GMP -> Host Memory, parameters
	// Replicate parameters. Supports multiple pairs of p and q
	// Faster than accessing only one pair from Global Memory

	for (i = 0; i < NUM_MSGS; i++) {
		mpz_export(dp_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, dp);
		mpz_export(dq_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, dq);
		mpz_export(p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, p);
		mpz_export(q_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, q);
		mpz_export(rp_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, rp);
		mpz_export(r2p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, r2p);
		mpz_export(rq_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, rq);
		mpz_export(r2q_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, r2q);
		mpz_export(q_i_p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, q_i_p);
	}

	for (i = 0; i < NUM_MSGS; i++) {
		p0inv_h[2 * i] = mpz_get_ui(p0inv);
		q0inv_h[2 * i] = mpz_get_ui(q0inv);
	}
	mpz_tdiv_q_2exp(p0inv, p0inv, 32);
	mpz_tdiv_q_2exp(q0inv, q0inv, 32);
	for (i = 0; i < NUM_MSGS; i++) {
		p0inv_h[2 * i + 1] = mpz_get_ui(p0inv);
		q0inv_h[2 * i + 1] = mpz_get_ui(q0inv);
	}

	// Start iteration 

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (j = 0; j < NUM_ITER; j++) {

		printf("Starting iteration %d\n", j);

		printf("Generating data...\n"); fflush(stdout);

		// Set with RND inputs
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_urandomb(msg_gmp[i], rnd_state, BIT_SIZE);
			mpz_mod(msg_p_gmp[i], msg_gmp[i], p);
			mpz_mod(msg_q_gmp[i], msg_gmp[i], q);
		}

		// GMP -> Host Memory, messages
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_export(msg_p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, msg_p_gmp[i]);
			mpz_export(msg_q_h + (i * BLOCK_SIZE), NULL, -1, sizeof(unsigned int), -1, 0, msg_q_gmp[i]);
		}

		printf("Starting kernel...\n"); fflush(stdout);

		// Start timing the event
		cudaEventRecord(start, 0);

		// Copy Host Memory -> Device Memory
		cudaMemcpy(msg_p_d, msg_p_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(msg_q_d, msg_q_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		cudaMemcpy(dp_d, dp_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(dq_d, dq_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		cudaMemcpy(p_d, p_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(q_d, q_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		cudaMemcpy(rp_d, rp_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(r2p_d, r2p_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		cudaMemcpy(rq_d, rq_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(r2q_d, r2q_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		cudaMemcpy(p0inv_d, p0inv_h, sizeof(unsigned int) * 2 * NUM_MSGS, cudaMemcpyHostToDevice);
		cudaMemcpy(q0inv_d, q0inv_h, sizeof(unsigned int) * 2 * NUM_MSGS, cudaMemcpyHostToDevice);

		cudaMemcpy(q_i_p_d, q_i_p_h, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);

		// Invoke Kernel
		cudaFuncSetCacheConfig(ModExp1, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(ModExp2, cudaFuncCachePreferShared);
		ModExp1 << <NUM_MSGS, BLOCK_SIZE >> > (msg_p_d, msg_q_d, p_d, q_d, p0inv_d, q0inv_d, dp_d, dq_d, rp_d, r2p_d, rq_d, r2q_d, smsg_p_d, smsg_q_d, e_p_d, e_q_d);
		ModExp2 << <NUM_MSGS, BLOCK_SIZE >> > (msg_p_d, msg_q_d, p_d, q_d, p0inv_d, q0inv_d, dp_d, dq_d, rp_d, r2p_d, rq_d, r2q_d, smsg_p_d, smsg_q_d, e_p_d, e_q_d);
		cudaThreadSynchronize();

		// Copy Device Memory -> Host Memory
		cudaMemcpy(smsg_p_h, smsg_p_d, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyDeviceToHost);
		cudaMemcpy(smsg_q_h, smsg_q_d, sizeof(unsigned int) * TOTAL_NUM_THREADS, cudaMemcpyDeviceToHost);

		// Host Memory -> GMP
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_import(smsg_gmp_gpu_p[i], BLOCK_SIZE, -1, sizeof(unsigned int), -1, 0, smsg_p_h + (i * BLOCK_SIZE));
			mpz_mod(smsg_gmp_gpu_p[i], smsg_gmp_gpu_p[i], p);
			mpz_import(smsg_gmp_gpu_q[i], BLOCK_SIZE, -1, sizeof(unsigned int), -1, 0, smsg_q_h + (i * BLOCK_SIZE));
			mpz_mod(smsg_gmp_gpu_q[i], smsg_gmp_gpu_q[i], q);
		}

		// Host Memory -> GMP
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_import(smsg_gmp_gpu_p[i], BLOCK_SIZE, -1, sizeof(unsigned int), -1, 0, smsg_p_h + (i * BLOCK_SIZE));
			mpz_mod(smsg_gmp_gpu_p[i], smsg_gmp_gpu_p[i], p);
			mpz_import(smsg_gmp_gpu_q[i], BLOCK_SIZE, -1, sizeof(unsigned int), -1, 0, smsg_q_h + (i * BLOCK_SIZE));
			mpz_mod(smsg_gmp_gpu_q[i], smsg_gmp_gpu_q[i], q);
		}

		// Stop timing the event
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timeExp, start, stop);

		delay = timeExp / 1000;   // GPU timer has resolution ms
		accum_delay += delay;     // accumulate delay for average
		throughput = NUM_MSGS / delay;
		printf("{%d, %d, %1.12g (sec), %1.12g (ops/sec)}\n", (int)BLOCK_SIZE, (int)NUM_MSGS, delay, throughput);
		fflush(stdout);

		printf("Verifying results on CPU...\n"); fflush(stdout);

		mpz_t aux1;
		mpz_init(aux1);
		mpz_setbit(aux1, 1024);

		mpz_t temp1;
		mpz_init(temp1);
		mpz_t temp2;
		mpz_init(temp2);
		mpz_t temp3;
		mpz_init(temp3);

		// Compare results between GMP and GPU
		for (i = 0; i < NUM_MSGS; i++) {

			mpz_powm(smsg_gmp_p[i], msg_p_gmp[i], dp, p);
			mpz_powm(smsg_gmp_q[i], msg_q_gmp[i], dq, q);

			if ((((mpz_cmp(smsg_gmp_gpu_p[i], smsg_gmp_p[i]) != 0))) | (mpz_cmp(smsg_gmp_gpu_q[i], smsg_gmp_q[i]) != 0)) {
				printf("Not equal!\n");
				printf("Problem in block No: %d\n", i);

				printf("msg_p_gmp  [%d] = ", i);
				mpz_out_str(stdout, 16, msg_p_gmp[i]); printf("\n");
				printf("dp [%d] = ", i);
				mpz_out_str(stdout, 16, dp); printf("\n");
				printf("p  [%d] = ", i);
				mpz_out_str(stdout, 16, p); printf("\n");
				printf("r2p[%d] = ", i);
				mpz_out_str(stdout, 16, r2p); printf("\n");
				printf("smsg_gmp_p  [%d] = ", i);
				mpz_out_str(stdout, 16, smsg_gmp_p[i]); printf("\n");
				printf("smsg_gmp_p + p[%d] = ", i);
				mpz_add(temp1, smsg_gmp_p[i], p);
				mpz_out_str(stdout, 16, temp1); printf("\n");
				printf("temp2 =  ");
				mpz_out_str(stdout, 16, temp2); printf("\n");
				mpz_add(temp3, temp2, p);
				printf("temp2 + p       = ");
				mpz_out_str(stdout, 16, temp3); printf("\n");
				printf("smsg_gpu_p  [%d] = ", i);
				mpz_out_str(stdout, 16, smsg_gmp_gpu_p[i]); printf("\n");

				printf("\n");

				printf("msg_q_gmp  [%d] = ", i);
				mpz_out_str(stdout, 16, msg_q_gmp[i]); printf("\n");
				printf("dq [%d] = ", i);
				mpz_out_str(stdout, 16, dq); printf("\n");
				printf("q [%d] = ", i);
				mpz_out_str(stdout, 16, q); printf("\n");
				printf("r2q[%d] = ", i);
				mpz_out_str(stdout, 16, r2q); printf("\n");
				printf("smsg_gmp_q  [%d] = ", i);
				mpz_out_str(stdout, 16, smsg_gmp_q[i]); printf("\n");
				printf("smsg_gmp_q+q[%d] = ", i);
				mpz_add(temp1, smsg_gmp_q[i], q);
				mpz_out_str(stdout, 16, temp1); printf("\n");
				printf("smsg_gpu_q  [%d] = ", i);
				mpz_out_str(stdout, 16, smsg_gmp_gpu_q[i]); printf("\n");
				printf("\n");

				printf("------------------------------------------------------------------------------------------------\n\n");

				printf("msg_p_gmp  [%d] = ", i);
				fprint_mpz(msg_p_gmp[i]);
				printf("dp [%d] = ", i);
				fprint_mpz(dp);
				printf("p  [%d] = ", i);
				fprint_mpz(p);
				printf("r2p[%d] = ", i);
				fprint_mpz(r2p);
				printf("smsg_gmp_p  [%d] = ", i);
				fprint_mpz(smsg_gmp_p[i]);
				printf("smsg_gmp_p+p[%d] = ", i);
				mpz_add(temp1, smsg_gmp_p[i], p);
				fprint_mpz(temp1);
				printf("temp2 =  ");
				fprint_mpz(temp2);
				printf("smsg_gpu_p  [%d] = ", i);
				fprint_mpz(smsg_gmp_gpu_p[i]);
				printf("\n");

				printf("msg_q_gmp  [%d] = ", i);
				fprint_mpz(msg_q_gmp[i]);
				printf("dq [%d] = ", i);
				fprint_mpz(dq);
				printf("q [%d] = ", i);
				fprint_mpz(q);
				printf("r2q[%d] = ", i);
				fprint_mpz(r2q);
				printf("smsg_gmp_q  [%d] = ", i);
				fprint_mpz(smsg_gmp_q[i]);
				printf("smsg_gmp_q+q[%d] = ", i);
				mpz_add(temp1, smsg_gmp_q[i], q);
				fprint_mpz(temp1);
				printf("smsg_gpu_q  [%d] = ", i);
				fprint_mpz(smsg_gmp_gpu_q[i]);
				printf("\n");

				abort();
			}
		}
		printf("Test passed!\n\n");
	}

	accum_delay /= NUM_ITER;
	throughput = NUM_MSGS / accum_delay;
	printf("{%d, %d, %1.12g (sec), %1.12g (ops/sec)}\n", (int)BLOCK_SIZE, (int)NUM_MSGS, accum_delay, throughput);
	fflush(stdout);

	// Release memory
	free(msg_p_h); free(msg_q_h);
	free(dp_h); free(dq_h);
	free(p_h);  free(q_h);
	free(rp_h); free(r2p_h);
	free(rq_h); free(r2q_h);
	free(p0inv_h); free(q0inv_h);
	free(smsg_p_h); free(smsg_q_h);

	for (i = 0; i < NUM_MSGS; i++) {
		mpz_clear(msg_p_gmp[i]); mpz_clear(msg_q_gmp[i]);
		mpz_clear(smsg_gmp_p[i]); mpz_clear(smsg_gmp_q[i]);
		mpz_clear(smsg_gmp_gpu_p[i]); mpz_clear(smsg_gmp_gpu_q[i]);
	}

	mpz_clear(dp); mpz_clear(dq);
	mpz_clear(p); mpz_clear(q); mpz_clear(pq);
	mpz_clear(rp); mpz_clear(r2p);
	mpz_clear(rq); mpz_clear(r2q);

	mpz_clear(p0inv);
	mpz_clear(q0inv);

	mpz_clear(e);
	mpz_clear(d);

	mpz_clear(pm1);
	mpz_clear(qm1);

	mpz_clear(phi);

	mpz_clear(temp_gmp);
	mpz_clear(pow2_ws);

	free(msg_p_gmp); free(msg_q_gmp);
	free(smsg_gmp_p); free(smsg_gmp_q);
	free(smsg_gmp_gpu_p); free(smsg_gmp_gpu_q);

	cudaFree(msg_p_d); cudaFree(msg_q_d);
	cudaFree(dp_d); cudaFree(dq_d);
	cudaFree(smsg_p_d); cudaFree(smsg_q_d);
	cudaFree(p_d); cudaFree(q_d);
	cudaFree(p0inv_d); cudaFree(q0inv_d);
	cudaFree(rp_d); cudaFree(r2p_d);
	cudaFree(rq_d); cudaFree(r2q_d);
	cudaFree(smsg_p_d); cudaFree(smsg_q_d);
	cudaFree(e_p_d); cudaFree(e_q_d);
	cudaFree(q_i_p_d);

	return EXIT_SUCCESS;

}*/