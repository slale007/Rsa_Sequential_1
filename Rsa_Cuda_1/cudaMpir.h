#ifndef __CUDAMPIR_H__

#include <stdio.h>
#include "..\mpir-3.0.0\mpir.h"

extern void RightShiftBlocks(mpz_t result, mpz_t inputNumber, int shift);

extern void Multiplication(mpz_t result, mpz_t first, mpz_t second);

#endif