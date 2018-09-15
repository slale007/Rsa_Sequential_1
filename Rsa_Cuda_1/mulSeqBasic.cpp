#include"mulSeqBasic.h"




void mulSeqBasic(mpz_t result, mpz_t first, mpz_t second)
{
	int tmpResultLength = (first->_mp_size + second->_mp_size) * 4;
	unsigned long long * tmpResult = (unsigned long long int *)malloc((1 + tmpResultLength) * sizeof(unsigned long long int));

	mpz_init(result);
	result->_mp_size = tmpResultLength / 4;
	result->_mp_alloc = result->_mp_size;
	result->_mp_d = (unsigned long long int *)malloc(result->_mp_size * sizeof(unsigned long long int));

	for (int i = 0; i < tmpResultLength+1; i++) {
		tmpResult[i] = 0;
	}

	unsigned short* firstShort = (unsigned short *)first->_mp_d;
	unsigned short* secondShort = (unsigned short *)second->_mp_d;

	for (int i = 0; i < first->_mp_size * 4; i++) {
		for (int k = 0; k < second->_mp_size * 4; k++) {
			tmpResult[i + k] += (unsigned long long int )firstShort[i] * secondShort[k];
		}
	}

	unsigned long long int carry = 0;
	unsigned long long int tmp = 0;
	for (int i = 0; i < tmpResultLength; i++) {
		tmp = tmpResult[i] + carry;
		carry = tmp >> 16;
		tmpResult[i] = tmp & 0xffff;
	}

	if (carry != 0) {
		tmpResult[tmpResultLength] = carry;
		tmpResultLength++;
	}

	for (int k = 0; k < (tmpResultLength+3) / 4; k++) {
		result->_mp_d[k] = 0;
		result->_mp_d[k] |= tmpResult[4 * k];
		result->_mp_d[k] |= (unsigned long long)tmpResult[4 * k + 1] << 16;
		result->_mp_d[k] |= (unsigned long long)tmpResult[4 * k + 2] << 2 * 16;
		result->_mp_d[k] |= (unsigned long long)tmpResult[4 * k + 3] << 3 * 16;
	}	

	if (result->_mp_d[result->_mp_size - 1] == 0) {
		result->_mp_size = result->_mp_size - 1;
	}
}










void RightShiftSeqBasic(mpz_t result, mpz_t a, unsigned int shiftCnt) {
	// For CUDA I will use another implementation this one is good for sequential implementation
	mpz_tdiv_q_2exp(result, a, shiftCnt);
}

void LeftShiftSeqBasic(mpz_t result, mpz_t a, unsigned int shiftCnt) {
	// For CUDA I will use another implementation this one is good for sequential implementation
	mpz_mul_2exp(result, a, shiftCnt);
}

void BarretModularReductionV2(mpz_t result, mpz_t xxx, mpz_t modul, mpz_t factor) {
	mpz_t tmp1, q1, q2, q3, r1, r2, r3;
	mpz_inits(tmp1, q1, q2, q3, r1, r2, r3, NULL);
	mpz_t tempNull;
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);
	mpz_t XxX;
	mpz_init(XxX);
	mpz_add(XxX, xxx, tempNull);
	
	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int k = 64 * modul->_mp_size - 64 + indexpom+1; // ok

	RightShiftSeqBasic(q1, XxX, k-1);
	//mulSeqBasic(q2, q1, factor);
	mpz_mul(q2, q1, factor);
	RightShiftSeqBasic(q2, q2, k+1);
	// mpz_add_ui(q2, q2, 2);

	//mulSeqBasic(r2, q2, modul);
	mpz_mul(r2, q2, modul);

	mpz_sub(result, XxX, r2);

	while (mpz_cmp(result, modul) >= 0)
	{
		mpz_sub(result, result, modul);
	}
}







void BarretModularReduction(mpz_t result, mpz_t x, mpz_t modul) {
	mpz_t factor, tmp1, q1, q2, q3, r1, r2, r3;
	mpz_inits(factor, tmp1, q1, q2, q3, r1, r2, r3, NULL);

	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int k = (64 * modul->_mp_size - 64 + indexpom + 1 ) / 2; // ok

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