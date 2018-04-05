#ifndef __CUDAMPIR_H__

#include <stdio.h>
#include "..\mpir-3.0.0\lib\x64\Debug\mpir.h"

extern void RightShift(mpz_t result, mpz_t inputNumber, mpz_t shift);

extern void RightShift2(mpz_t result, mpz_t inputNumber, int shift);

#endif