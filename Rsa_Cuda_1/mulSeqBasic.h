#ifndef __MULSEQBASIC_H__

#include <stdio.h>
#include "..\mpir-3.0.0\mpir.h"

extern void mulSeqBasic(mpz_t results, mpz_t a, mpz_t b);

extern void BarretModularReduction(mpz_t result, mpz_t x, mpz_t modul);
extern void BarretModularReductionV2(mpz_t result, mpz_t x, mpz_t modul, mpz_t factor);

#endif