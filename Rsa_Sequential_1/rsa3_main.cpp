/////////////////////////////////////
// My implementation of Montgomery //
/////////////////////////////////////

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <string.h>

#include "stdafx.h"
#include "rsa1.h"
#include "timeutil.h"
#define BASE 64

using namespace std;

void MontgomeryModularMultiplication(mpz_t res, mpz_t x, mpz_t yyy, mpz_t modul, unsigned long long int mprim)
{
	mpz_t temp1;
	mpz_t temp2;
	mpz_t tempNull;
	mpz_t uim;
	mpz_t xiy;
	mpz_t XI;
	mpz_t UI;
	mpz_init(temp1);
	mpz_init(temp2);
	mpz_init(tempNull);
	mpz_init(uim);
	mpz_init(xiy);
	mpz_init(XI);
	mpz_init(UI);
	mpz_init(res);
	
	for (int i = 0; i < modul->_mp_size; i++) {
		unsigned long long int xi;
		if (i < x->_mp_size)
		{
			xi = x->_mp_d[i];
		}
		else
		{
			xi = 0;
		}

		unsigned long long int ui;

		if (res->_mp_size == 0)
		{
			ui = ((xi * yyy->_mp_d[0]) * mprim);
		}
		else
		{
			ui = ((res->_mp_d[0] + xi * yyy->_mp_d[0]) * mprim);
		}

		// little data type conversion
		mpz_add_ui(XI, tempNull, xi);

		// xiy = xi * y
		mpz_mul(xiy, yyy, XI);

		// little data type conversion
		mpz_add_ui(UI, tempNull, ui);

		// uim = ui * modul
		mpz_mul(uim, UI, modul);

		// temp1 = xiy + uim
		mpz_add(temp1, xiy, uim);

		// temp2 = res + temp1
		mpz_add(temp2, res, temp1);

		mpz_tdiv_q_2exp(res, temp2, 64);

	}

	if (mpz_cmp(res, modul) >= 0)
	{
		mpz_sub(temp1, res, modul);
		mpz_add(res, tempNull, temp1);
	}
}

void MontgomeryModularMultiplicationBinary(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, unsigned long long int mprim)
{
	mpz_t temp1;
	mpz_t temp2;
	mpz_t tempNull;
	mpz_t uim;
	mpz_t xiy;
	mpz_t XI;
	mpz_t UI;
	mpz_init(temp1);
	mpz_init(temp2);
	mpz_init(tempNull);
	mpz_init(uim);
	mpz_init(xiy);
	mpz_init(XI);
	mpz_init(UI);
	mpz_init(res);

	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int index = 64 * modul->_mp_size - 64 + indexpom;

	for (int i = 0; i < index; i++) {
		unsigned long long int xi;
		if (i < xxx->_mp_size * 64)
		{
			xi = (xxx->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) != 0 ? 1 : 0;
		}
		else
		{
			xi = 0;
		}

		unsigned long long int ui;

		if (res->_mp_size == 0)
		{
			ui = ((xi * (yyy->_mp_d[0] & 1)) * mprim) % 2;
		}
		else
		{
			ui = (((res->_mp_d[0] & 1) + xi * (yyy->_mp_d[0] & 1)) * mprim) % 2;
		}

		// little data type conversion
		mpz_add_ui(XI, tempNull, xi);

		// xiy = xi * y
		mpz_mul(xiy, yyy, XI);

		// little data type conversion
		mpz_add_ui(UI, tempNull, ui);

		// uim = ui * modul
		mpz_mul(uim, UI, modul);

		// temp1 = xiy + uim
		mpz_add(temp1, xiy, uim);

		// temp2 = res + temp1
		mpz_add(temp2, res, temp1);

		mpz_tdiv_q_2exp(res, temp2, 1);
	}

	if (mpz_cmp(res, modul) >= 0)
	{
		mpz_sub(temp1, res, modul);
		mpz_add(res, tempNull, temp1);
	}
}

void MontgomeryModularEponentiation(mpz_t A, mpz_t x, mpz_t exponent, mpz_t modul)
{
	mpz_t R;
	mpz_t RsquareMod;
	mpz_t Rsquare;
	mpz_t tempNull;
	mpz_init(R);
	mpz_init(RsquareMod);
	mpz_init(Rsquare);
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);

	mpz_add_ui(R, R, 1);
	mpz_mul_2exp(R, R, 64 * modul->_mp_size);
	mpz_mul_2exp(Rsquare, R, 1);
	mpz_mod(RsquareMod, Rsquare, modul);

	mpz_t mprim;
	mpz_init(mprim);
	mpz_t base;
	mpz_init(base);
	mpz_add_ui(base, tempNull, 1);
	mpz_mul_2exp(base, base, 64);

	mpz_t min1;
	mpz_init(min1);

	mpz_set_ui(min1, 0);
	mpz_sub_ui(min1, min1, 1);
	mpz_powm(mprim, modul, min1, base);
	mpz_sub(mprim, base, mprim);

	mpz_clears(min1, base, NULL);

	mpz_t xline;
	mpz_init(xline);
	MontgomeryModularMultiplication(xline, x, RsquareMod, modul, mprim->_mp_d[0]);

	mpz_mod(A, R, modul);

	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (exponent->_mp_d[exponent->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}

	int index = 64 * exponent->_mp_size - 64 + indexpom;
	for (int i = index; i >= 0; i--) {
		mpz_t AA;
		mpz_t AAAA;
		mpz_init(AA);
		mpz_init(AAAA);
		mpz_add(AA, tempNull, A);
		mpz_add(AAAA, tempNull, A);
		MontgomeryModularMultiplication(A, AA, AAAA, modul, mprim->_mp_d[0]);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MontgomeryModularMultiplication(A, AA, xline, modul, mprim->_mp_d[0]);
		}
	}

	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, A);
	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplication(A, AAA, one, modul, mprim->_mp_d[0]);
}

// Full Checked times: 1
void MontgomeryModularEponentiationBinary(mpz_t A, mpz_t x, mpz_t exponent, mpz_t modul)
{
	mpz_t R;
	mpz_t RsquareMod;
	mpz_t Rsquare;
	mpz_t tempNull;
	mpz_init(R);
	mpz_init(RsquareMod);
	mpz_init(Rsquare);
	mpz_init(tempNull);
	mpz_add_ui(tempNull, tempNull, 0);
	mpz_add_ui(R, R, 1);

	int indexpom0 = 0;
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom0 = i;
			break;
		}
	}
	int index0 = 64 * modul->_mp_size - 64 + indexpom0;

	mpz_mul_2exp(R, R, index0 + 1);
	mpz_mul_2exp(Rsquare, R, 1);
	mpz_mod(RsquareMod, Rsquare, modul);

	mpz_t mprim;
	mpz_init(mprim);
	mpz_t base;
	mpz_init(base);
	mpz_set_ui(base, 2);

	mpz_t min1;
	mpz_init(min1);
	mpz_set_ui(min1, 0);
	mpz_sub_ui(min1, min1, 1);
	mpz_powm(mprim, modul, min1, base);
	mpz_sub(mprim, base, mprim);

	mpz_clears(min1, base, NULL);

	// all above stuff is checked

	mpz_t xline;
	mpz_init(xline);
	MontgomeryModularMultiplicationBinary(xline, x, RsquareMod, modul, mprim->_mp_d[0]);

	mpz_mod(A, R, modul);

	int indexpom = 0;
	for (int i = 63; i >= 0; i--) {
		if (exponent->_mp_d[exponent->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom = i;
			break;
		}
	}
	int index = 64 * exponent->_mp_size - 64 + indexpom; // ok
	for (int i = index; i >= 0; i--) {
		mpz_t AA;
		mpz_t AAAA;
		mpz_init(AA);
		mpz_init(AAAA);
		mpz_add(AA, tempNull, A);
		mpz_add(AAAA, tempNull, A);
		MontgomeryModularMultiplicationBinary(A, AA, AAAA, modul, mprim->_mp_d[0]);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MontgomeryModularMultiplicationBinary(A, AA, xline, modul, mprim->_mp_d[0]);
		}
	}

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, A);

	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplicationBinary(A, AAA, one, modul, mprim->_mp_d[0]);
}


void rsac_encryptV2(public_key *pub, const char *message, size_t m_len, char **cryptedMessage, size_t *c_len)
{
	mpz_t m_int, c_int, c_int2, c_int3;
	mpz_inits(m_int, c_int, c_int2, c_int3, NULL);
	mpz_import(
		m_int, m_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, message);

	// rsac_encrypt_internal(pub, m_int, c_int);
	MontgomeryModularEponentiationBinary(/*cripted*/c_int,/* message */ m_int, /*exponent*/ pub->e, /*modul*/ pub->n);

	*cryptedMessage = (char*)mpz_export(NULL, c_len, 1, 1, 1, 0, c_int);
	// mpz_clears(m_int, c_int, NULL);
}

int test_rsac_string_encrypt_decrypt3() {
	std::clock_t start = std::clock();
	char m[] = "stop slacking off.";
	size_t c_len, m_len = strlen(m), result_len;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	int fail = 0;
	public_key* pub = (public_key*)calloc(sizeof(public_key), 1);
	private_key* priv = (private_key*)calloc(sizeof(private_key), 1);

	if (pub == NULL || priv == NULL) {
		printf("FAIL: rsac_string_encrypt_decrypt could not allocate public or private key struct\n");
		return 1;
	}

	int res = rsac_keygen(pub, priv);
	if (res != 0) {
		printf("FAIL: rsac_string_encrypt_decrypt rsac_keygen returned %d, expected 0\n", res);
		fail++;
	}

	printTime(start);
	start = std::clock();

    rsac_encryptV2(pub, m, m_len, c, &c_len);

	rsac_decrypt(priv, *c, c_len, m_result, &result_len);
	if (strlen(*m_result) != m_len || strncmp(m, *m_result, m_len) != 0) {
		printf("FAIL: rsac_string_encrypt_decrypt message did not match after encryption and decryption.\n");
		printf("expected '%s' but got '%s'\n", m, *m_result);
		fail++;
	}

	free(pub);
	free(priv);
	free(*c);
	free(*m_result);
	if (fail == 0) {
		printf("PASS: rsac_string_encrypt_decrypt\n");
	}

	printTime(start);
	return fail;
}



int mainsss() {
	int failures = 0;

	printf(" CHAR_BIT je: %d\n", CHAR_BIT);
	printf("Velicina char je: %d\n", sizeof(char));
	printf("Velicina mp_limb_t je: %d\n", sizeof(mp_limb_t));
	printf("Velicina unsigned long int je: %d\n", sizeof(unsigned long int));


	failures += test_rsac_string_encrypt_decrypt3();

	printf("%d failures\n", failures);
	return failures > 0;
}