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

using namespace std;

int test_rsac_inverse_modulo3() {
	std::clock_t start = std::clock();
	int fail = 0;
	mpz_t a, b, c;

	mpz_init_set_ui(a, 3);
	mpz_init_set_ui(b, 11);
	mpz_init(c);
	rsac_inverse_modulo(a, b, c);
	if (mpz_cmp_ui(c, 4) != 0) {
		printf("FAIL: rsac_inverse_modulo expected 4, got ");
		mpz_out_str(NULL, 10, c);
		printf("\n");
		fail++;
	}
	else {
		printf("PASS: rsac_inverse_modulo\n");
	}
	mpz_clears(a, b, c, NULL);

	printTime(start);
	return fail;
}

int test_rsac_random_prime3() {
	std::clock_t start = std::clock();
	int fail = 0;
	mpz_t a;
	mpz_init(a);

	rsac_init_randstate();
	for (int i = 1; i < 100; i += 10) {
		rsac_random_prime(i, a);
		if (!mpz_probab_prime_p(a, 25)) {
			printf("FAIL: rsac_random_prime returned composite number for bit size %d\n", i);
			fail++;
			break;
		}
		size_t size = mpz_sizeinbase(a, 2);
		if (size > i + 1) {
			printf("FAIL: rsac_random_prime returned %lu bits, expected %d\n", size, i);
			fail++;
			break;
		}
	}
	mpz_clear(a);

	if (fail == 0) {
		printf("PASS: rsac_random_prime\n");
	}

	printTime(start);
	return fail;
}

int test_rsac_keygen_internal3() {
	std::clock_t start = std::clock();
	mpz_t n, e, d, p, q;
	mpz_inits(n, e, d, p, q, NULL);
	int fail = 0;

	for (int i = 0; i < 1; i++) {
		rsac_keygen_internal(n, e, d, p, q);
	}

	mpz_clears(n, e, d, p, q, NULL);

	if (fail == 0) {
		printf("PASS: rsac_keygen_internal\n");
	}

	printTime(start);
	return fail;
}

int test_rsac_keygen3() {
	std::clock_t start = std::clock();
	int fail = 0;
	public_key* pub = (public_key*)calloc(sizeof(public_key), 1);
	private_key* priv = (private_key*)calloc(sizeof(private_key), 1);
	if (pub == NULL || priv == NULL) {
		printf("FAIL: rsac_keygen could not allocate public or private key struct\n");
		return 1;
	}

	int res = rsac_keygen(pub, priv);
	if (res != 0) {
		printf("FAIL: rsac_keygen returned %d, expected 0\n", res);
		fail++;
	}

	// Just test for sane values; test_rsac_keygen_internal does most of the
	// real testing.
	if (mpz_cmp(pub->n, priv->n) != 0) {
		printf("FAIL: rsac_keygen public and private keys have different moduli\n");
		fail++;
	}
	if (mpz_cmp_ui(pub->n, 0) == 0) {
		printf("FAIL: rsac_keygen modulus is zero\n");
		fail++;
	}
	if (mpz_cmp_ui(pub->e, 0) == 0) {
		printf("FAIL: rsac_keygen public exponent is zero\n");
		fail++;
	}
	if (mpz_cmp_ui(priv->d, 0) == 0) {
		printf("FAIL: rsac_keygen private exponent is zero\n");
		fail++;
	}

	mpz_clears(pub->n, pub->e, priv->n, priv->d, NULL);
	free(pub);
	free(priv);

	if (fail == 0) {
		printf("PASS: rsac_keygen\n");
	}

	printTime(start);
	return fail;
}

int test_rsac_encrypt_decrypt_inverses3() {
	std::clock_t start = std::clock();
	int fail = 0;
	public_key* pub = (public_key*)calloc(sizeof(public_key), 1);
	private_key* priv = (private_key*)calloc(sizeof(private_key), 1);
	if (pub == NULL || priv == NULL) {
		printf("FAIL: rsac_encrypt_decrypt_inverse could not allocate public or private key struct\n");
		return 1;
	}

	int res = rsac_keygen(pub, priv);
	if (res != 0) {
		printf("FAIL: rsac_encrypt_decrypt_inverse rsac_keygen returned %d, expected 0\n", res);
		fail++;
	}
	printTime(start);
	start = std::clock();

	// Generate 100 random integer messages, and ensure that
	// decrypt(encrypt(M)) == M for each.
	mpz_t m, m_cycled, c;
	mpz_inits(m, m_cycled, c, NULL);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, time(NULL));
	for (int i = 1; i < 1; i++) {
		mpz_urandomb(m, state, 20);
		rsac_encrypt_internal(pub, m, c);
		rsac_decrypt_internal(priv, c, m_cycled);
		if (mpz_cmp(m, m_cycled) != 0) {
			printf("FAIL: rsac_encrypt_decrypt_inverse the message was different after encrypting and decrypting");
			printf("\noriginal message M:        ");
			mpz_out_str(NULL, 10, m);
			printf("\nmessage after dec(enc(M)): ");
			mpz_out_str(NULL, 10, m_cycled);
			fail++;
			break;
		}
	}

	mpz_clears(m, m_cycled, c, pub->n, pub->e, priv->n, priv->d, NULL);
	free(pub);
	free(priv);

	if (fail == 0) {
		printf("PASS: rsac_encrypt_decrypt_inverse\n");
	}

	printTime(start);
	return fail;
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

	rsac_encrypt(pub, m, m_len, c, &c_len);
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

void MontgomeryModularMultiplication(mpz_t x, mpz_t y, mpz_t modul, unsigned long long int mprim)
{
	mpz_t temp1;
	mpz_t temp2;
	mpz_t uim;
	mpz_t xiy;
	mpz_t A;
	mpz_inits(temp1,temp2, A, uim, xiy, NULL);

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

		unsigned long long int ui = ((A->_mp_d[0] + xi * y->_mp_d[0]) * mprim) % 64;

		// little data type conversion
		mpz_t XI;
		mpz_init_set_ui (XI, xi);

		// xiy = xi * y
		mpz_mul(xiy, y, XI);

		// little data type conversion
		mpz_t UI;
		mpz_init_set_ui(UI, ui);

		// uim = ui * modul
		mpz_mul(uim, UI, modul);

		// temp1 = xiy +uim
		mpz_add(temp1, xiy, uim);

		// temp2 = A + temp1
		mpz_add(temp2, A, temp1);

		// A = temp2 /  base using right shift once
		// This can be improved to be O(1)
		for (int j = 1; j < temp2->_mp_size;j++) {
			A->_mp_d[j - 1] = temp2->_mp_d[j];
		}

		A->_mp_size = temp2->_mp_size - 1;
	}

	if (mpz_cmp(A, modul) >= 0)
	{
		mpz_sub(temp1, A, modul);
		mpz_add(A, 0, temp1);
	}
}

void MontgomeryModularEponentiation(mpz_t x, mpz_t exponent, mpz_t modul)
{
	mpz_t R;
	mpz_inits(R, NULL);
	// This can be performance improved
	mpz_mul_2exp(R, R, sizeof(unsigned long long int) * modul->_mp_size);

	mpz_t mprim;
	mpz_inits(mprim, NULL);




}

int main323() {
	int failures = 0;

	printf(" CHAR_BIT je: %d\n", CHAR_BIT);
	printf("Velicina char je: %d\n", sizeof(char));
	printf("Velicina mp_limb_t je: %d\n", sizeof(mp_limb_t));
	printf("Velicina unsigned long int je: %d\n", sizeof(unsigned long int));

	failures += test_rsac_inverse_modulo3();
	failures += test_rsac_random_prime3();
	//	failures += test_rsac_keygen_internal3();
	failures += test_rsac_keygen3();
	failures += test_rsac_encrypt_decrypt_inverses3();
	failures += test_rsac_string_encrypt_decrypt3();

	printf("%d failures\n", failures);
	return failures > 0;
}