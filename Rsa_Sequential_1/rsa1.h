#ifndef __RSA1_H__

#include <stdio.h>
#include "..\mpir-3.0.0\lib\x64\Debug\mpir.h"

typedef struct {
	mpz_t n, e;
} public_key;

typedef struct {
	mpz_t n, d;
} private_key;

// Returns 0 if successful.
// out: pub, priv
int rsac_keygen(public_key *pub, private_key *priv);

// in: pub, m, m_len, out: c, c_len
void rsac_encrypt(public_key *pub, const char *m, size_t m_len, char **c, size_t *c_len);

// in: priv, c, c_len, out: m, m_len
void rsac_decrypt(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len);


void rsac_init_randstate();

// in: bit_size, out: x
void rsac_random_prime(unsigned int bit_size, mpz_t x);

// out: e
void rsac_public_exponent(mpz_t e);

// in: a, b, out: x
void rsac_inverse_modulo(mpz_t a, mpz_t b, mpz_t x);

// (n, e) is public key, (n, d) is private key. p and q are provided for
// testing. Consumers of the API should use rsac_keygen(pub, priv).
// Returns 0 if successful.
// out: n, e, d, p, q
int rsac_keygen_internal(mpz_t n, mpz_t e, mpz_t d, mpz_t p, mpz_t q);

// in: pub, m, out: c
void rsac_encrypt_internal(public_key *pub, mpz_t m, mpz_t c);

// in: priv, c, out: m
void rsac_decrypt_internal(private_key *priv, mpz_t c, mpz_t m);
#endif