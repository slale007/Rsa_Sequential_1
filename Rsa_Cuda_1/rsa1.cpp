#include "stdafx.h"
#include "rsa1.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <string.h>

using namespace std;

// This is used for RSA keygen 1024 2048 3072 4096 8192
// Note: For "2048-key" RSA you need to generate two primes of 1024 bits. "The RSA key size is not the size of private key eksponent. It is number of bits in the public modulus (which is known as N).
#define PQ_PRIME_SIZE_BITS 1024

#define NUMBER_OF_ROUNDS_IN_KEY_GENERATION 2

// This will be used later, in keygen 2
#define MODULUS_SIZE 8192                   /* This is the number of bits we want in the modulus */
#define BLOCK_SIZE (MODULUS_SIZE/8)         /* This is the size of a block that gets en/decrypted at once */
#define BUFFER_SIZE ((MODULUS_SIZE/8) / 2)  /* This is the number of bytes in n and p */

gmp_randstate_t state;

#pragma region RSA_Keygen

void rsac_public_exponent(mpz_t e)
{
	mpz_set_ui(e, 65537);
}

// let result = a^-1 mod b
void rsac_inverse_modulo(mpz_t a, mpz_t b, mpz_t result)
{
	mpz_invert(result, a, b);
}

void rsac_init_randstate()
{
	gmp_randinit_default(state);
	gmp_randseed_ui(state, time(NULL));
}

void rsac_random_prime(unsigned int bit_size, mpz_t x)
{
	mpz_urandomb(x, state, bit_size);
	mpz_nextprime(x, x);
}

int rsaKeyGenerationInternal(mpz_t n, mpz_t e, mpz_t d, mpz_t p, mpz_t q)
{
	// t1 and t2 are temp variables
	mpz_t phi, t1, t2;
	mpz_inits(t1, t2, phi, NULL);

	mpz_set_ui(d, 0);
	rsac_public_exponent(e);

	int rounds;
	for (rounds = 0; mpz_cmp_ui(d, 0) == 0 && rounds < NUMBER_OF_ROUNDS_IN_KEY_GENERATION; rounds++) {
		rsac_init_randstate();
		rsac_random_prime(PQ_PRIME_SIZE_BITS, p);
		rsac_random_prime(PQ_PRIME_SIZE_BITS, q);

		// the public modulus n := p * q
		mpz_mul(n, p, q);

		// phi := (p - 1)(q - 1). Phi is the number of integers less than n that are
		// relatively prime to n.
		mpz_sub_ui(t1, p, 1);
		mpz_sub_ui(t2, q, 1);
		mpz_mul(phi, t1, t2);

		rsac_inverse_modulo(e, phi, d);
	}

	mpz_clears(t1, t2, phi, NULL);

	if (rounds == NUMBER_OF_ROUNDS_IN_KEY_GENERATION) {
		return -1;
	}
	return 0;
}

void rsaKeyGeneration(public_key *publicKey, private_key *privateKey)
{
	mpz_t n, e, d, p, q;
	mpz_inits(n, e, d, p, q, NULL);

	int success = rsaKeyGenerationInternal(n, e, d, p, q);

	if (success != 0) {
		printf("FAIL: rsaKeyGeneration returned %d, expected 0\n", success);
		return;
	}

	mpz_init_set(publicKey->n, n);
	mpz_init_set(publicKey->e, e);
	mpz_init_set(privateKey->n, n);
	mpz_init_set(privateKey->d, d);
	mpz_clears(n, e, d, p, q, NULL);
}


void rsaKeyGeneration2(public_key *publicKey, private_key *privateKey)
{
	char buf[BUFFER_SIZE];
	int i;
	mpz_t phi; mpz_init(phi);
	mpz_t tmp1; mpz_init(tmp1);
	mpz_t tmp2; mpz_init(tmp2);

	srand(time(NULL));

	/* Insetead of selecting e st. gcd(phi, e) = 1; 1 < e < phi, lets choose e
	* first then pick p,q st. gcd(e, p-1) = gcd(e, q-1) = 1 */
	// We'll set e globally.  I've seen suggestions to use primes like 3, 17 or 
	// 65537, as they make coming calculations faster.  Lets use 65537.
	mpz_set_ui(privateKey->e, 65537);

	/* Select p and q */
	/* Start with p */
	// Set the bits of tmp randomly
	for (i = 0; i < BUFFER_SIZE; i++)
		buf[i] = rand() % 0xFF;
	// Set the top two bits to 1 to ensure int(tmp) is relatively large
	buf[0] |= 0xC0;
	// Set the bottom bit to 1 to ensure int(tmp) is odd (better for finding primes)
	buf[BUFFER_SIZE - 1] |= 0x01;
	// Interpret this char buffer as an int
	mpz_import(tmp1, BUFFER_SIZE, 1, sizeof(buf[0]), 0, 0, buf);
	// Pick the next prime starting from that random number
	mpz_nextprime(privateKey->p, tmp1);
	/* Make sure this is a good choice*/
	mpz_mod(tmp2, privateKey->p, privateKey->e);        /* If p mod e == 1, gcd(phi, e) != 1 */
	while (!mpz_cmp_ui(tmp2, 1))
	{
		mpz_nextprime(privateKey->p, privateKey->p);    /* so choose the next prime */
		mpz_mod(tmp2, privateKey->p, privateKey->e);
	}

	/* Now select q */
	do {
		for (i = 0; i < BUFFER_SIZE; i++)
			buf[i] = rand() % 0xFF;
		// Set the top two bits to 1 to ensure int(tmp) is relatively large
		buf[0] |= 0xC0;
		// Set the bottom bit to 1 to ensure int(tmp) is odd
		buf[BUFFER_SIZE - 1] |= 0x01;
		// Interpret this char buffer as an int
		mpz_import(tmp1, (BUFFER_SIZE), 1, sizeof(buf[0]), 0, 0, buf);
		// Pick the next prime starting from that random number
		mpz_nextprime(privateKey->q, tmp1);
		mpz_mod(tmp2, privateKey->q, privateKey->e);
		while (!mpz_cmp_ui(tmp2, 1))
		{
			mpz_nextprime(privateKey->q, privateKey->q);
			mpz_mod(tmp2, privateKey->q, privateKey->e);
		}
	} while (mpz_cmp(privateKey->p, privateKey->q) == 0); /* If we have identical primes (unlikely), try again */

										  /* Calculate n = p x q */
	mpz_mul(privateKey->n, privateKey->p, privateKey->q);

	/* Compute phi(n) = (p-1)(q-1) */
	mpz_sub_ui(tmp1, privateKey->p, 1);
	mpz_sub_ui(tmp2, privateKey->q, 1);
	mpz_mul(phi, tmp1, tmp2);

	/* Calculate d (multiplicative inverse of e mod phi) */
	if (mpz_invert(privateKey->d, privateKey->e, phi) == 0)
	{
		mpz_gcd(tmp1, privateKey->e, phi);
		printf("gcd(e, phi) = [%s]\n", mpz_get_str(NULL, 16, tmp1));
		printf("Invert failed\n");
	}

	/* Set public key */
	mpz_set(publicKey->e, privateKey->e);
	mpz_set(publicKey->n, privateKey->n);
}


#pragma endregion








#pragma region RSA_Algorithm

void rsac_encrypt_internal(public_key *pub, mpz_t m, mpz_t c)
{
	mpz_powm(c, m, pub->e, pub->n);
}

void rsac_decrypt_internal(private_key *priv, mpz_t c, mpz_t m)
{
	mpz_powm(m, c, priv->d, priv->n);
}

void rsac_encrypt(public_key *pub, const char *message, size_t m_len, char **cryptedMessage, size_t *c_len)
{
	mpz_t m_int, c_int;
	mpz_inits(m_int, c_int, NULL);
	mpz_import(
		m_int, m_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, message);
	rsac_encrypt_internal(pub, m_int, c_int);
	*cryptedMessage = (char*)mpz_export(NULL, c_len, 1, 1, 1, 0, c_int);
	mpz_clears(m_int, c_int, NULL);
}

void rsac_decrypt(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int;
	mpz_inits(m_int, c_int, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);
	rsac_decrypt_internal(priv, c_int, m_int);
	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, NULL);
}
#pragma endregion

