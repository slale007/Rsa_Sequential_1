// GPU code for performing RSA-2048
// Author: Marcelo Kaihara
// Date: 15-03-2011
/*
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <gmp.h>

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

typedef unsigned long long int uint64_cu

// Inline PTX instructions-----------------------------------------------
static inline __device__ uint addc32(uint a, uint b) {
	uint c;
	asm("addc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ uint64_cu addc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("addc.u64 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ uint addc_cc32(uint a, uint b) {
	uint c;
	asm("addc.cc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ uint64_cu addc_cc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("addc.cc.u64 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ uint add_cc32(uint a, uint b) {
	uint c;
	asm("add.cc.u32 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

static inline __device__ uint64_cu add_cc64(uint64_cu a, uint64_cu b) {
	uint64_cu c;
	asm("add.cc.u64 %0, %1, %2;" : "=r" (c) : "r" (a), "r" (b));
	return c;
}

#define __mul_hi32(c, x, y) c = __umulhi(x,y)
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
// Compute: (regOut3, regOut2, regOut1, regOut0) = (regB1, regB0) * (regA1, regA0)
// Temp registers: regTemp4, regTemp3, regTemp2, regTemp1, regTemp0
#define LWMul32(regOut3, regOut2, regOut1, regOut0, regB1, regB0, regA1, regA0, regTemp4, regTemp3, regTemp2, regTemp1, regTemp0)  { \
    __mul_hi32 (regTemp3, regA0, regB0);					\
    regTemp1 = regA0 * regB1;						\
    __mul_hi32 (regTemp2, regA0, regB1);					\
    __mul_hi32 (regTemp0, regA1, regB0);					\
    __mul_hi32 (regTemp4, regA1, regB1);					\
    regOut0 = regA0 * regB0;						\
    regOut1 = add_cc32 (regTemp1, regTemp3);				\
    regOut2 = addc_cc32 (regTemp0, regTemp2);				\
    regOut3 = addc32 (regTemp4, 0);					\
    regTemp1 = regA1 * regB0;						\
    regTemp2 = regA1 * regB1;						\
    regOut1 = add_cc32 (regOut1, regTemp1);				\
    regOut2 = addc_cc32 (regOut2, regTemp2);				\
    regOut3 = addc32 (regOut3, 0);					\
  }

// Long Word Multiplication
// Compute: (regOut1, regOut0) = regB * regA
#define LWMul64(regOut1, regOut0, regB, regA)  { \
    __mul_hi64 (regOut1, regA, regB);					\
    regOut0 = regA0 * regB1;						\
  }


// Lower part of Long Word Multiplication
// Compute: (regOut1, regOut0) = (regB1, regB0) * (regA1, regA0) mod (2^64)
// Temp registers: regTemp2, regTemp1, regTemp0
#define LWHMul32(regOut1, regOut0, regB1, regB0, regA1, regA0, regTemp2, regTemp1, regTemp0) { \
    __mul_hi32 (regTemp2, regA0, regB0);					\
    regTemp1 = regA0 * regB1;						\
    regTemp0 = regA1 * regB0;						\
    regOut0 = regA0 * regB0;						\
    regOut1 = regTemp2 + regTemp1 + regTemp0;				\
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
__device__ void  MMDouble(
	uint64_cu * local_carries,
	uint64_cu * local_temp,
	const uint64_cu * local_A,
	const uint64_cu * local_B
	const uint64_cu * local_M
	const uint64_cu * local_MM
	const uint64_cu reg_M0inv)
{
	int i;

	uint64_cu mask;

	uint64_cu qM;

	uint64_cu regTemp1_2;
	uint64_cu regTemp1_1;
	uint64_cu regTemp1;
	uint64_cu regTemp0;

	uint64_cu reg;
	uint64_cu regC;

	uint64_cu regA = local_A[threadIdx.x];

	uint64_cu regP = local_M[threadIdx.x];

	uint64_cu regB = local_B[0];

	uint64_cu regShared1 = 0;

	uint64_cu regCarries1 = 0;

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
		local_temp[(threadIdx.x + 1) & MASK32] = add_cc(local_temp[(threadIdx.x + 1) & MASK32], local_carries[threadIdx.x]);
		local_carries[(threadIdx.x + 1) & MASK32] = addc(0, 0);

	}

	if (threadIdx.x > 30) {
		local_carries[threadIdx.x] += regCarries1;
	}

	// If yet carry left, then result is larger than P
	if (local_carries[LAST_ITEM]) {

		mask = -1;

		// Accumulate -P
		local_temp[threadIdx.x] = add_cc(local_temp[threadIdx.x], local_MM[threadIdx.x] & mask);
		regCarries1 = addc(local_carries[threadIdx.x], 0);

		if (threadIdx.x < 30) {
			local_carries[threadIdx.x] = regCarries1;
		}
		else {
			local_carries[threadIdx.x] = 0;
		}

		for (i = 15; i > 0; i--) {

			local_temp[((threadIdx.x + 2) & MASK32)] = add_cc(local_temp[((threadIdx.x + 2) & MASK32)], local_carries[threadIdx.x]);
			local_carries[((threadIdx.x + 1) & MASK32)] = addc(0, 0);
		}

		if (threadIdx.x > 30) {
			local_carries[threadIdx.x] += (regCarries1 + mask);
		}
	}
}











// **********************************************************************************
// Modular Exponentiation ***********************************************************
// **********************************************************************************

#define __exp_step(index) {					\
    local_e_p [threadIdx.x] = global_e_p [blockDim.x * ( WEXP_NUM_PRECOMP * blockIdx.x + (index)) + threadIdx.x]; \
    for (j = WEXP_WSIZE - 1; j >= 0; j--) {				\
      MMDouble (carries_1_p, shared_1_p, local_temp_vec_p, local_temp_vec_p, local_p, local_pp, p0inv); \
      local_temp_vec_p [threadIdx.x] = shared_1_p [threadIdx.x];	\
    }									\
    MMDouble (carries_1_p, shared_1_p, local_temp_vec_p, tempA, local_p, local_pp, p0inv); \
    local_temp_vec_p [threadIdx.x] = shared_1_p [threadIdx.x];		\
  }




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




__global__ void __launch_bounds__(32, 8) ModExp1(
	const uint *message,
	const uint *modul,
	const uint *Modul0INV,
	const uint *RP,
	const uint *R2P, // R na kvadrat - koristii se samo za prebacivanje u Montgomery formu 
	uint *global_e_p)// 
{

	// Vectors data
	__shared__ uint local_temp_vec_p[BLOCK_SIZE];

	__shared__ uint local_base_p[BLOCK_SIZE];

	__shared__ uint local_p[BLOCK_SIZE];

	__shared__ uint local_r2p[BLOCK_SIZE];

	// Temp Vectors required for MM
	__shared__ uint carries_1_p[BLOCK_SIZE];
	__shared__ uint shared_1_p[BLOCK_SIZE];


	uint modul0INV, q0inv;

	if ((threadIdx.x & 1) == 0) {
		modul0INV = Modul0INV[2 * blockIdx.x + 1];
		q0inv = Modul0INV[2 * blockIdx.x];
	}
	else {
		modul0INV = Q0INV[2 * blockIdx.x + 1];
		q0inv = Q0INV[2 * blockIdx.x];
	}

	int i;

	local_p[threadIdx.x] = modul[blockDim.x * blockIdx.x + threadIdx.x];

	local_base_p[threadIdx.x] = message[blockDim.x * blockIdx.x + threadIdx.x];

	local_r2p[threadIdx.x] = R2P[blockDim.x * blockIdx.x + threadIdx.x];

	// Precompute -p and -q
	local_pp[threadIdx.x] = ~local_p[threadIdx.x];
	if (threadIdx.x == 0) { local_pp[threadIdx.x] |= 1; }  // p is assumed to be odd


														   // Transform operand to Montgomery representation------------------------------------------------------
	MMDouble(
		carries_1_p,
		shared_1_p,
		local_base_p,
		local_r2p,
		local_p,
		local_pp,
		modul0INV);

	local_base_p[threadIdx.x] = shared_1_p[threadIdx.x];  // here base_p is correct  

														  // Do the precomputations for Windows Exponentiation---------------------------------------------------
														  // ^ 0
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x) + threadIdx.x] = RP[blockDim.x * blockIdx.x + threadIdx.x];

	// ^ 1
	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 1) + threadIdx.x] = shared_1_p[threadIdx.x];

	// ^ 2
	MMDouble(carries_1_p, local_temp_vec_p,
		local_base_p,
		local_base_p,
		local_p,
		local_pp,
		modul0INV);

	global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + 2) + threadIdx.x] = local_temp_vec_p[threadIdx.x];

	for (i = 3; i < WEXP_NUM_PRECOMP; i++) {

		MMDouble(carries_1_p, shared_1_p,
			local_temp_vec_p,
			local_base_p,
			local_p,
			local_pp,
			modul0INV);

		local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];

		global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + i) + threadIdx.x] = shared_1_p[threadIdx.x];
	}

}



















__global__ void __launch_bounds__(32, 8) ModExp2(
	const uint *A_p,
	const uint *P,
	const uint *P0INV,
	const uint *DMP1,
	const uint *RP,
	const uint *R2P,
	uint *Z_p,
	uint *global_e_p)
{

	// Vectors data
	__shared__ uint local_temp_vec_p[BLOCK_SIZE];

	__shared__ uint local_base_p[BLOCK_SIZE];

	__shared__ uint local_p[BLOCK_SIZE];
	__shared__ uint local_pp[BLOCK_SIZE];

	__shared__ uint local_dmp1[BLOCK_SIZE];

	__shared__ uint local_r2p[BLOCK_SIZE];

	// Temp Vectors required for MM
	__shared__ uint carries_1_p[BLOCK_SIZE];
	__shared__ uint shared_1_p[BLOCK_SIZE];

	__shared__ uint shared_1_q[BLOCK_SIZE];
	__shared__ uint local_temp_carry_q[BLOCK_SIZE];

	// Vectors for precomputatin (Windows Exp)
	__shared__ uint local_e_p[BLOCK_SIZE];

	uint index1p;
	uint index2p;

	uint p0inv;

	if ((threadIdx.x & 1) == 0) {
		p0inv = P0INV[2 * blockIdx.x + 1];
	}
	else {
		p0inv = Q0INV[2 * blockIdx.x + 1];
	}

	int i;
	int j;

	local_p[threadIdx.x] = P[blockDim.x * blockIdx.x + threadIdx.x];

	local_base_p[threadIdx.x] = A_p[blockDim.x * blockIdx.x + threadIdx.x];

	local_r2p[threadIdx.x] = R2P[blockDim.x * blockIdx.x + threadIdx.x];

	local_dmp1[threadIdx.x] = DMP1[blockDim.x * blockIdx.x + threadIdx.x];

	// Precompute -p 
	local_pp[threadIdx.x] = ~local_p[threadIdx.x];
	if (threadIdx.x == 0) { local_pp[threadIdx.x] |= 1; }  // p is assumed to be odd

														   // Initialize vectors----------------------------------------------------------------------
	carries_1_p[threadIdx.x] = 0;
	shared_1_p[threadIdx.x] = 0;

	// Load data from global memory
	local_temp_vec_p[threadIdx.x] = RP[blockDim.x * blockIdx.x + threadIdx.x];


	//------------------------------------------------------------------------------------------
	for (i = LAST_ITEM; i >= 6; i -= 5) {

		index1p = local_dmp1[i];

		__exp_step(index1p >> 27);

		index2p = local_dmp1[i - 1];

		__exp_step((index1p >> 22) & 0x1F);
		__exp_step((index1p >> 17) & 0x1F);
		__exp_step((index1p >> 12) & 0x1F);
		__exp_step((index1p >> 7) & 0x1F);
		__exp_step((index1p >> 2) & 0x1F);

		__exp_step(((index1p << 3) | (index2p >> 29)) & 0x1F);
		__exp_step((index2p >> 24) & 0x1F);
		__exp_step((index2p >> 19) & 0x1F);
		__exp_step((index2p >> 14) & 0x1F);
		__exp_step((index2p >> 9) & 0x1F);

		index1p = local_dmp1[i - 2];

		__exp_step((index2p >> 4) & 0x1F);

		__exp_step(((index2p << 1) | (index1p >> 31)) & 0x1F);
		__exp_step((index1p >> 26) & 0x1F);
		__exp_step((index1p >> 21) & 0x1F);
		__exp_step((index1p >> 16) & 0x1F);
		__exp_step((index1p >> 11) & 0x1F);
		__exp_step((index1p >> 6) & 0x1F);

		index2p = local_dmp1[i - 3];

		__exp_step((index1p >> 1) & 0x1F);

		__exp_step(((index1p << 4) | (index2p >> 28)) & 0x1F);
		__exp_step((index2p >> 23) & 0x1F);
		__exp_step((index2p >> 18) & 0x1F);
		__exp_step((index2p >> 13) & 0x1F);
		__exp_step((index2p >> 8) & 0x1F);

		index1p = local_dmp1[i - 4];

		__exp_step((index2p >> 3) & 0x1F);

		__exp_step(((index2p << 2) | (index1p >> 30)) & 0x1F);
		__exp_step((index1p >> 25) & 0x1F);
		__exp_step((index1p >> 20) & 0x1F);
		__exp_step((index1p >> 15) & 0x1F);
		__exp_step((index1p >> 10) & 0x1F);
		__exp_step((index1p >> 5) & 0x1F);
		__exp_step((index1p) & 0x1F);

	}

	index1p = local_dmp1[i];

	__exp_step(index1p >> 27);
	__exp_step((index1p >> 22) & 0x1F);
	__exp_step((index1p >> 17) & 0x1F);
	__exp_step((index1p >> 12) & 0x1F);
	__exp_step((index1p >> 7) & 0x1F);

	index2p = local_dmp1[i - 1];

	__exp_step((index1p >> 2) & 0x1F);

	__exp_step(((index1p << 3) | (index2p >> 29)) & 0x1F);
	__exp_step((index2p >> 24) & 0x1F);
	__exp_step((index2p >> 19) & 0x1F);
	__exp_step((index2p >> 14) & 0x1F);
	__exp_step((index2p >> 9) & 0x1F);
	__exp_step((index2p >> 4) & 0x1F, );

	local_e_p[threadIdx.x] = global_e_p[blockDim.x * (WEXP_NUM_PRECOMP * blockIdx.x + (index2p & 0xF)) + threadIdx.x]; // load precomputed value in folded representation


	for (j = WEXP_WSIZE - 2; j >= 0; j--) {
		MMDouble(carries_1_p, shared_1_p,
			local_temp_vec_p,
			local_temp_vec_p,
			local_p,
			local_pp,
			p0inv);
		local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];
	}
	MMDouble(carries_1_p, shared_1_p,
		local_temp_vec_p,
		tempA,
		local_p,
		local_pp,
		p0inv);

	local_temp_vec_p[threadIdx.x] = shared_1_p[threadIdx.x];

	// Transform back from montgomery presentation to original representation
	if (threadIdx.x < 2) {
		local_r2p[threadIdx.x] = 0;
	}
	else {
		local_r2p[threadIdx.x] = 0;
	}

	MMDouble(carries_1_p, shared_1_p,
		local_temp_vec_p,
		local_r2p,
		local_p,
		local_pp,
		p0inv);

	// Store result in global memory
	Z_p[blockDim.x * blockIdx.x + threadIdx.x] = shared_1_p[threadIdx.x];
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
int mainr(int argc, char** argv)
{
	printf("Benchmark: RSA2048 on GPU\n");

	int i, j;

	float timeExp = 0;
	float delay = 0;
	float accum_delay = 0;
	float throughput = 0;

	cudaEvent_t start, stop;

	// GMP variables
	mpz_t * msg_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * message = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * result_GMP_host = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * result_from_GPU = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * smsg_gmp = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	mpz_t * smsg_gmp_gpu = (mpz_t *)malloc(sizeof(mpz_t) * NUM_MSGS);
	for (i = 0; i < NUM_MSGS; i++) {
		mpz_init(msg_gmp[i]);
		mpz_init(message[i]);
		mpz_init(result_GMP_host[i]);
		mpz_init(result_from_GPU[i]);
		mpz_init(smsg_gmp[i]);
		mpz_init(smsg_gmp_gpu[i]);
	}

	mpz_t temp_gmp;
	mpz_init(temp_gmp);

	gmp_randstate_t rnd_state;
	gmp_randinit_default(rnd_state);
	gmp_randseed_ui(rnd_state, 598182);


	// Host variables
	uint * message_host = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * dp_h = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * p_h = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * rp_h = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * r2p_h = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * p0inv_h = (uint *)malloc(sizeof(uint) * 2 * NUM_MSGS);
	uint * q_i_p_h = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);
	uint * pom1 = (uint *)malloc(sizeof(uint) * TOTAL_NUM_THREADS);


	// Device variables
	uint * message_GPU;
	uint * dp_d;
	uint * modul_GPU;
	uint * modulPrim_GPU;
	uint * rp_d;
	uint * r2p_d;
	uint * result_GPU;
	uint * e_p_d;
	uint * q_i_p_d;

	printf("Initializing device\n"); fflush(stdout);

	// Allocate vectors in device memory
	cudaMalloc((void **)&message_GPU, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&dp_d, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&modul_GPU, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&modulPrim_GPU, sizeof(uint) * 2 * NUM_MSGS);
	cudaMalloc((void **)&rp_d, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&r2p_d, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&result_GPU, sizeof(uint) * TOTAL_NUM_THREADS);
	cudaMalloc((void **)&e_p_d, sizeof(uint) * TOTAL_NUM_THREADS * WEXP_NUM_PRECOMP);
	cudaMalloc((void **)&q_i_p_d, sizeof(uint) * TOTAL_NUM_THREADS);

	// Generate parameters
	mpz_t modul, pq, pm1, phi, e, d, p_i_q, q_i_p, dp;
	mpz_t p0inv;
	mpz_t pow2_ws;
	mpz_t rp, r2p;

	mpz_init(modul);
	mpz_init(pq);

	do
	{
		mpz_urandomb(modul, rnd_state, BIT_SIZE / 2);
		mpz_setbit(modul, BIT_SIZE / 2 - 1);
		mpz_setbit(modul, BIT_SIZE / 2 - 2);
		mpz_setbit(modul, 0);

		mpz_gcd(pq, modul, q);
	} while (mpz_cmp_ui(pq, 1) != 0);

	mpz_mul(pq, modul, q);

	mpz_init_set_ui(e, RSA_EXP);
	mpz_init(d);
	mpz_init(pm1);
	mpz_init(phi);

	mpz_sub_ui(pm1, modul, 1);
	mpz_sub_ui(qm1, q, 1);
	mpz_mul(phi, pm1, qm1);
	if (mpz_invert(d, e, phi) == 0)
		abort();

	mpz_init(p_i_q);
	if (mpz_invert(p_i_q, modul, q) == 0)
		abort();

	mpz_init(q_i_p);
	if (mpz_invert(q_i_p, q, modul) == 0)
		abort();

	// sve jasno gore

	mpz_init(dp);
	mpz_init(dq);
	mpz_mod(dp, d, pm1);
	mpz_mod(dq, d, qm1);

	mpz_init(p0inv);
	mpz_init(pow2_ws);
	mpz_setbit(pow2_ws, WORD_SIZE);
	mpz_neg(temp_gmp, modul);
	mpz_invert(p0inv, temp_gmp, pow2_ws);

	mpz_init(rp);
	mpz_set_ui(rp, 0);
	mpz_setbit(rp, 1024);
	mpz_mod(rp, rp, modul);
	mpz_init(r2p);
	mpz_mul(r2p, rp, rp);
	mpz_mod(r2p, r2p, modul);

	// GMP -> Host Memory, parameters
	// Replicate parameters. Supports multiple pairs of p and q
	// Faster than accessing only one pair from Global Memory

	for (i = 0; i < NUM_MSGS; i++) {
		mpz_export(dp_h + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, dp);
		mpz_export(p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, modul);
		mpz_export(rp_h + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, rp);
		mpz_export(r2p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, r2p);
		mpz_export(q_i_p_h + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, q_i_p);
	}

	for (i = 0; i < NUM_MSGS; i++) {
		p0inv_h[2 * i] = mpz_get_ui(p0inv);
	}
	mpz_tdiv_q_2exp(p0inv, p0inv, 32); // this is right ( -> ) shift 
	for (i = 0; i < NUM_MSGS; i++) {
		p0inv_h[2 * i + 1] = mpz_get_ui(p0inv);
	}

	// Start iteration 

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (j = 0; j < NUM_ITER; j++) {

		printf("Starting iteration %d\n", j);

		printf("Generating data...\n"); fflush(stdout);

		// Set with RaNDom inputs
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_urandomb(msg_gmp[i], rnd_state, BIT_SIZE);
			mpz_mod(message[i], msg_gmp[i], modul);
		}

		// GMP -> Host Memory, messages
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_export(message_host + (i * BLOCK_SIZE), NULL, -1, sizeof(uint), -1, 0, message[i]);
		}

		printf("Starting kernel...\n"); fflush(stdout);

		// Start timing the event
		cudaEventRecord(start, 0);

		// Copy Host Memory -> Device Memory
		cudaMemcpy(message_GPU, message_host, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(dp_d, dp_h, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(modul_GPU, p_h, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(rp_d, rp_h, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(r2p_d, r2p_h, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);
		cudaMemcpy(modulPrim_GPU, p0inv_h, sizeof(uint) * 2 * NUM_MSGS, cudaMemcpyHostToDevice);
		cudaMemcpy(q_i_p_d, q_i_p_h, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyHostToDevice);


		__global__ void __launch_bounds__(32, 8) ModExp1(
			const uint *message,
			const uint *modul,
			const uint *Modul0INV,
			const uint *RP,
			const uint *R2P,
			uint *global_e_p)


			// Invoke Kernel
			cudaFuncSetCacheConfig(ModExp1, cudaFuncCachePreferShared);
		cudaFuncSetCacheConfig(ModExp2, cudaFuncCachePreferShared);
		ModExp1 << <NUM_MSGS, BLOCK_SIZE >> > (
			message_GPU, // random message mod p
			modul_GPU, // modul
			modulPrim_GPU, // this is (-m0)^-1(mod B)
			rp_d, // 1000...00000 mod(p)
			r2p_d, // rp_d ^2 mod(p)
			result_GPU, // probably this is result
			e_p_d
			);
		ModExp2 << <NUM_MSGS, BLOCK_SIZE >> > (
			message_GPU,
			p_d,
			modulPrim_GPU,
			dp_d, /// ovo je zbog CRT-a generalno treba da bude samo d - stepen
			rp_d,
			r2p_d,
			result_GPU,
			e_p_d
			);

		cudaThreadSynchronize();

		// Copy Device Memory -> Host Memory
		cudaMemcpy(pom1, result_GPU, sizeof(uint) * TOTAL_NUM_THREADS, cudaMemcpyDeviceToHost);

		// Host Memory -> GMP
		for (i = 0; i < NUM_MSGS; i++) {
			mpz_import(result_from_GPU[i], BLOCK_SIZE, -1, sizeof(uint), -1, 0, pom1 + (i * BLOCK_SIZE));
			mpz_mod(result_from_GPU[i], result_from_GPU[i], p);
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

			mpz_powm(result_GMP_host[i], message[i], dp, p);

			if (((
				(mpz_cmp(result_from_GPU[i], result_GMP_host[i]) != 0)))) {
				printf("Not equal!\n");
				printf("Problem in block No: %d\n", i);

				printf("message  [%d] = ", i);
				mpz_out_str(stdout, 16, message[i]); printf("\n");
				printf("dp [%d] = ", i);
				mpz_out_str(stdout, 16, dp); printf("\n");
				printf("p  [%d] = ", i);
				mpz_out_str(stdout, 16, p); printf("\n");
				printf("r2p[%d] = ", i);
				mpz_out_str(stdout, 16, r2p); printf("\n");
				printf("result_GMP_host  [%d] = ", i);
				mpz_out_str(stdout, 16, result_GMP_host[i]); printf("\n");
				printf("result_GMP_host + p[%d] = ", i);
				mpz_add(temp1, result_GMP_host[i], p);
				mpz_out_str(stdout, 16, temp1); printf("\n");
				printf("temp2 =  ");
				mpz_out_str(stdout, 16, temp2); printf("\n");
				mpz_add(temp3, temp2, p);
				printf("temp2 + p       = ");
				mpz_out_str(stdout, 16, temp3); printf("\n");
				printf("smsg_gpu_p  [%d] = ", i);
				mpz_out_str(stdout, 16, result_from_GPU[i]); printf("\n");
				printf("\n");

				printf("------------------------------------------------------------------------------------------------\n\n");

				printf("message  [%d] = ", i);
				fprint_mpz(message[i]);
				printf("dp [%d] = ", i);
				fprint_mpz(dp);
				printf("p  [%d] = ", i);
				fprint_mpz(p);
				printf("r2p[%d] = ", i);
				fprint_mpz(r2p);
				printf("result_GMP_host  [%d] = ", i);
				fprint_mpz(result_GMP_host[i]);
				printf("result_GMP_host+p[%d] = ", i);
				mpz_add(temp1, result_GMP_host[i], p);
				fprint_mpz(temp1);
				printf("temp2 =  ");
				fprint_mpz(temp2);
				printf("smsg_gpu_p  [%d] = ", i);
				fprint_mpz(result_from_GPU[i]);
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
	free(message_host);
	free(dp_h);
	free(p_h);
	free(rp_h); free(r2p_h);
	free(p0inv_h);
	free(pom1);

	for (i = 0; i < NUM_MSGS; i++) {
		mpz_clear(message[i]);
		mpz_clear(result_GMP_host[i]);
		mpz_clear(result_from_GPU[i]);
	}

	mpz_clear(dp);
	mpz_clear(p); mpz_clear(pq);
	mpz_clear(rp); mpz_clear(r2p);
	mpz_clear(p0inv);
	mpz_clear(e);
	mpz_clear(d);
	mpz_clear(pm1);
	mpz_clear(phi);
	mpz_clear(temp_gmp);
	mpz_clear(pow2_ws);
	free(message);
	free(result_GMP_host);
	free(result_from_GPU);

	cudaFree(message_GPU);
	cudaFree(dp_d);
	cudaFree(result_GPU);
	cudaFree(p_d);
	cudaFree(p0inv_d);
	cudaFree(rp_d); cudaFree(r2p_d);
	cudaFree(e_p_d);
	cudaFree(q_i_p_d);

	return EXIT_SUCCESS;

}*/