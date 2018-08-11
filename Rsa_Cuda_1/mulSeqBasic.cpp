#include"mulSeqBasic.h"

void mulSeqBasic(mpz_t result, mpz_t a, mpz_t b)
{
	int tmpResultLength = (a->_mp_size + b->_mp_size) * 4;
	unsigned long long * tmpResult = (unsigned long long int *)malloc((1 + tmpResultLength) * sizeof(unsigned long long int));

	mpz_init(result);
	result->_mp_size = tmpResultLength / 4;
	result->_mp_alloc = result->_mp_size;
	result->_mp_d = (unsigned long long int *)malloc(result->_mp_size * sizeof(unsigned long long int));

	for (int i = 0; i < tmpResultLength; i++) {
		tmpResult[i] = 0;
	}

	unsigned short* first = (unsigned short *)a->_mp_d;
	unsigned short* second = (unsigned short *)b->_mp_d;

	for (int i = 0; i < a->_mp_size * 4; i++) {
		for (int k = 0; k < b->_mp_size * 4; k++) {
			tmpResult[i + k] += first[i] * second[k];
		}
	}

	unsigned long long int carry = 0;
	unsigned long long int tmp = 0;
	int i;
	for (i = 0; i < tmpResultLength; i++) {
		tmp = tmpResult[i] + carry;
		carry = tmp >> 16;
		tmpResult[i] = tmp & 0xffff;
	}

	if (carry != 0) {
		tmpResult[i] = carry;
	}

	for (int k = 1; k < tmpResultLength / 4; k++) {
		result->_mp_d[k] = 0;
		result->_mp_d[k] |= tmpResult[4 * k];
		result->_mp_d[k] |= tmpResult[4 * k + 1] << 16;
		result->_mp_d[k] |= tmpResult[4 * k + 2] << 2 * 16;
		result->_mp_d[k] |= tmpResult[4 * k + 3] << 3 * 16;
	}	
}


void RightShiftSeqBasic(mpz_t result, mpz_t a, unsigned int shiftCnt) {
	// because for CUDA I will use another implementation this one is good for sequential implementation
	mpz_tdiv_q_2exp(result, a, shiftCnt);
}

void LeftShiftSeqBasic(mpz_t result, mpz_t a, unsigned int shiftCnt) {
	// because for CUDA I will use another implementation this one is good for sequential implementation
	mpz_mul_2exp(result, a, shiftCnt);
}

// This is precomputation
void calcualateBarrettFactor(mpz_t factor, mpz_t modul, int k) {

	mpz_t twoPower;
	mpz_init(twoPower);
	mpz_add_ui(twoPower, twoPower, 1);
	mpz_mul_2exp(twoPower, twoPower, 2 * k);

	mpz_div(factor, twoPower, modul);
}

void BarretModularReduction(mpz_t result, mpz_t x, mpz_t modul) {
	mpz_t factor, tmp1, q1, q2, q3, r1, r2, r3 ;
	mpz_inits(factor, tmp1, q1, q2, q3, r1, r2, r3, NULL);

	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int k = 64 * modul->_mp_size - 64 + indexpom + 1; // ok

	calcualateBarrettFactor(factor, modul, k);

	RightShiftSeqBasic(q1, x, k - 1);
	
	mulSeqBasic(q2, q1, factor);

	RightShiftSeqBasic(q3, q2, k + 1);

	// modul
	RightShiftSeqBasic(tmp1, x, k + 1);
	LeftShiftSeqBasic(tmp1, tmp1, k + 1);
	mpz_sub(r1, x, tmp1); 

	mulSeqBasic(r2, q3, modul);

	// modul
	RightShiftSeqBasic(tmp1, r2, k + 1);
	LeftShiftSeqBasic(tmp1, tmp1, k + 1);
	mpz_sub(r3, r2, tmp1);


	mpz_sub(result, r1, r3);

	if (mpz_cmp(result, modul) >= 0)
	{
		mpz_sub(result, result, modul);
	}
}