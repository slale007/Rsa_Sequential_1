/////////////////////////////////////
// My implementation of Montgomery //
// Failed                          //
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

// This one is correct!
void MontgomeryModularMultiplicationV4(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, mpz_t mprim, mpz_t R)
{
	mpz_t t;
	mpz_init(t);
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

	mpz_mul(t, xxx, yyy);
	mpz_mul(tmp1, t, mprim);
	mpz_mod(tmp2, tmp1, R);
	mpz_mul(tmp3, tmp2, modul);
	mpz_add(tmp4, t, tmp3);


	int index = 0;
	for (int i = 63; i >= 0; i--) {
		if (R->_mp_d[R->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			index = i;
			break;
		}
	}

	index = 64 * R->_mp_size - 64 + index;

	mpz_tdiv_q_2exp(u, tmp4, index);

	mpz_div(slowU, tmp4, R);
	if (mpz_cmp(u, slowU) != 0) {
		cout<<endl << "Fatal error" << endl;
	}

	// all above is good

	// step 3.
	if (mpz_cmp(u, modul) >= 0)
	{
		mpz_sub(res, u, modul);
	}
	else {
		mpz_add_ui(res, u, 0); // ok
	}
}

void MontgomeryModularEponentiationV4(mpz_t res, mpz_t xxx, mpz_t exponent, mpz_t modul)
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
	for (int i = 63; i >= 0; i--) {
		if (modul->_mp_d[modul->_mp_size - 1] & (((unsigned long long int)1) << i)) {
			indexpom0 = i;
			break;
		}
	}
	int index0 = 64 * modul->_mp_size - 64 + indexpom0;

	mpz_mul_2exp(RR, RR, index0 + 1);
	mpz_mul_2exp(Rsquare, RR, 1);
	mpz_mod(RsquareMod, Rsquare, modul);
	mpz_mod(RMod, RR, modul);

	// above is correct

	mpz_t mprim;
	mpz_init(mprim);
	mpz_t mprim2;
	mpz_init(mprim2);
	mpz_t base;
	mpz_init(base);
	mpz_set_ui(base, 2);

	mpz_t min1;
	mpz_t onne;
	mpz_init(onne);
	mpz_add_ui(onne, onne, 1);
	mpz_init(min1);
	mpz_sub(min1, tempNull, onne);
	mpz_powm(mprim, modul, min1, RR);
    mpz_invert(mprim2, modul, RR);

	if (mpz_cmp(mprim, mprim2) != 0) {
		cout << "Fatal error0" << endl;
	}

	mpz_sub(mprim, RR, mprim);
	mpz_sub(mprim2, tempNull, mprim2);
	mpz_mod(mprim2, mprim2, RR);

	if (mpz_cmp(mprim, mprim2) != 0) {
		cout << endl<<"Fatal error01" << endl;
	}

	// all above stuff is checked

	mpz_t xline;
	mpz_init(xline);
	mpz_t xline2pom;
	mpz_init(xline2pom);
	mpz_t xline2;
	mpz_init(xline2);
	mpz_mul(xline2pom, xxx, RR);
	mpz_mod(xline, xline2pom, modul);

	 // MontgomeryModularMultiplicationV4(xline, xxx, RsquareMod, modul, mprim, RR);

	if (mpz_cmp(xline, xline2) != 0) {
		cout << endl << "Fatal error02" << endl;
	}

	mpz_mod(res, RR, modul);

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
		mpz_add(AA, tempNull, res);
		mpz_add(AAAA, tempNull, res);



		MontgomeryModularMultiplicationV4(res, AA, AAAA, modul, mprim, RR);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			mpz_add(AA, tempNull, res);
			MontgomeryModularMultiplicationV4(res, AA, xline, modul, mprim, RR);
		}
		mpz_clears(AA, AAAA, NULL);
	}

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, res);

	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplicationV4(res, AAA, one, modul, mprim, RR);
}



void rsac_encryptV4(public_key *pub, const char *message, size_t m_len, char **cryptedMessage, size_t *c_len)
{
	mpz_t m_int, c_int, c_int2, c_int3;
	mpz_inits(m_int, c_int, c_int2, c_int3, NULL);
	mpz_import(
		m_int, m_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, message);

	// rsac_encrypt_internal(pub, m_int, c_int);
	MontgomeryModularEponentiationV4(/*cripted*/c_int,/* message */ m_int, /*exponent*/ pub->e, /*modul*/ pub->n);

	*cryptedMessage = (char*)mpz_export(NULL, c_len, 1, 1, 1, 0, c_int);
	// mpz_clears(m_int, c_int, NULL);
}

int test_rsac_string_encrypt_decrypt4() {
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

	rsac_encryptV4(pub, m, m_len, c, &c_len);

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



int main() {
	int failures = 0;

	printf(" CHAR_BIT je: %d\n", CHAR_BIT);
	printf("Velicina char je: %d\n", sizeof(char));
	printf("Velicina mp_limb_t je: %d\n", sizeof(mp_limb_t));
	printf("Velicina unsigned long int je: %d\n", sizeof(unsigned long int));
	printf("Velicina unsigned long long int je: %d\n", sizeof(unsigned long long int));

	failures += test_rsac_string_encrypt_decrypt4();

	printf("%d failures\n", failures);
	return failures > 0;
}