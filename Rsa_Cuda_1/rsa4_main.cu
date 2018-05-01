/////////////////////////////////////
// My implementation of Montgomery //
/////////////////////////////////////

#include "..\Rsa_Sequential_1\stdafx.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <string.h>

#include "..\Rsa_Cuda_1\rsa1.h"
#include "..\Rsa_Sequential_1\timeutil.h"
#include "..\Rsa_Cuda_1\customFunctions.h"
#include "cudaMpir.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BASE 64

using namespace std;

clock_t globalTime0;
clock_t globalTime1;
clock_t globalTime2;
clock_t globalTime3;
clock_t globalTime4;
clock_t globalTime5;
clock_t globalTime6;
clock_t globalTime7;
clock_t globalTime8;
clock_t start;

// This one is correct!
void MontgomeryModularMultiplicationV4(mpz_t res, mpz_t xxx, mpz_t yyy, mpz_t modul, mpz_t mprim, mpz_t R, int index)
{
	mpz_t t;
	mpz_init(t);
	mpz_t t2;
	mpz_init(t2);
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
	mpz_t u2;
	mpz_init(u2);
	mpz_t slowU;
	mpz_init(slowU);

    start = clock();

	mpz_mul(t, xxx, yyy);
	//Multiplication(t2, xxx, yyy);

	if (mpz_cmp(t, t2) != 0) {
		/*cout << std::hex << "Mul--- Fatal Error" << endl;
		/*cout << "Good value: "<<endl;
		for (int i = 0; i < t->_mp_size; i++) {
			cout << t->_mp_d[i] << endl;
		}
		cout <<  "Bad value: " << endl;
		for (int i = 0; i < t2->_mp_size; i++) {
			cout << t2->_mp_d[i] << endl;
		}*/
	}

	globalTime0 += clock() - start;
	start = clock();

	mpz_mul(tmp1, t, mprim);

	globalTime1 += clock() - start;
	start = clock();

	// mpz_mod(tmp2, tmp1, R);
	mpz_tdiv_q_2exp(tmp2, tmp1, index);
	mpz_mul_2exp(tmp2, tmp2, index);
	mpz_sub(tmp2, tmp1, tmp2);

	globalTime4 += clock() - start;
	start = clock();

	mpz_mul(tmp3, tmp2, modul);

	globalTime5 += clock() - start;
	start = clock();

	mpz_add(tmp4, t, tmp3);

	globalTime6 += clock() - start;
	start = clock();

    mpz_tdiv_q_2exp(u, tmp4, index);
	// RightShiftBlocks(u, tmp4, index/64);

	if (mpz_cmp(u, u2) != 0) {
		// cout << "Mod--- Fatal Error" << endl;
	}
	else {
		// cout << "Mod--- All Right" << endl;
	}

	globalTime7 += clock() - start;
	start = clock();

	// step 3.
	if (mpz_cmp(u, modul) >= 0)
	{
		mpz_sub(res, u, modul);
	}
	else {
		mpz_add_ui(res, u, 0); // ok
	}

	globalTime8 += clock() - start;
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

	mpz_mul_2exp(RR, RR, modul->_mp_size * 64);
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

	mpz_sub(mprim, RR, mprim);


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
		// cout << endl << "Fatal error02" << endl;
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

	int indexRR = 64 * RR->_mp_size - 64;

	for (int i = index; i >= 0; i--) {
		MontgomeryModularMultiplicationV4(res, res, res, modul, mprim, RR, indexRR);
		if (exponent->_mp_d[i / 64] & (((unsigned long long int)1) << (i % 64))) {
			MontgomeryModularMultiplicationV4(res, res, xline, modul, mprim, RR, indexRR);
		}
	}

	cout << "Global time 0: " << globalTime0 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 1: " << globalTime1 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 2: " << globalTime2 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 3: " << globalTime3 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 4: " << globalTime4 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 5: " << globalTime5 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 6: " << globalTime6 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 7: " << globalTime7 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
	cout << "Global time 8: " << globalTime8 / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;

	// all above stuff is checked
	mpz_t one;
	mpz_t AAA;
	mpz_init(one);
	mpz_init(AAA);
	mpz_add(AAA, tempNull, res);

	mpz_add_ui(one, one, 1);
	MontgomeryModularMultiplicationV4(res, AAA, one, modul, mprim, RR, indexRR);
}

void rsac_encryptV4(public_key *pub, const char *message, size_t m_len, char **cryptedMessage, size_t *c_len)
{
	mpz_t m_int, c_int, c_int2, c_int3;
	mpz_inits(m_int, c_int, c_int2, c_int3, NULL);
	mpz_import(m_int, m_len, /* MS word first */ 1, /* bytes per word */ 1, /* big-endian */ 1, /* skip bits */ 0, message);

	clock_t startTime = std::clock();
	MontgomeryModularEponentiationV4(/*cripted*/c_int,/* message */ m_int, /*exponent*/ pub->e, /*modul*/ pub->n);
	cout << "Montgomery realization: ";
	printTime(startTime);


	startTime = std::clock();
	rsac_encrypt_internal(pub, m_int, c_int2);
	cout << "Mpir realization: ";
	printTime(startTime);



	*cryptedMessage = (char*)mpz_export(NULL, c_len, 1, 1, 1, 0, c_int);
	// mpz_clears(m_int, c_int, NULL);
}

void rsac_decryptV4(private_key *priv, const char *c, size_t c_len, char **m, size_t *m_len)
{
	mpz_t m_int, c_int, m_int2;
	mpz_inits(m_int, c_int, m_int2, NULL);
	mpz_import(
		c_int, c_len, /* MS word first */ 1, /* bytes per word */ 1,
		/* big-endian */ 1, /* skip bits */ 0, c);


	clock_t startTime = std::clock();
	MontgomeryModularEponentiationV4(/*cripted*/m_int,/* message */ c_int, /*exponent*/ priv->d, /*modul*/ priv->n);
	cout << "Montgomery realization: ";
	printTime(startTime);

	// Mpir realization of powm
	startTime = std::clock();
	rsac_decrypt_internal(priv, c_int, m_int);
	cout << "Mpir realization: ";
	printTime(startTime);


	*m = (char*)mpz_export(NULL, m_len, 1, 1, 1, 0, m_int);
	mpz_clears(m_int, c_int, m_int2, NULL);
}

int test_rsac_string_encrypt_decrypt4() {
	char message[] = "Stop slacking off.Stop slacking off.Stop slacking off.Stop slacking off.";
	size_t c_len, m_len = strlen(message), result_len;
	char **c = (char**)calloc(sizeof(char *), 1);
	char **m_result = (char**)calloc(sizeof(char *), 1);
	int fail = 0;
	public_key* pub = (public_key*)calloc(sizeof(public_key), 1);
	private_key* priv = (private_key*)calloc(sizeof(private_key), 1);

	if (pub == NULL || priv == NULL) {
		printf("FAIL: rsac_string_encrypt_decrypt could not allocate public or private key struct\n");
		return 1;
	}

	printf("\n_______________________Key generation_______________________\n\n");

	// Initialize public key
	mpz_init(pub->n);
	mpz_init(pub->e);
	// Initialize private key
	mpz_init(priv->n);
	mpz_init(priv->e);
	mpz_init(priv->d);
	mpz_init(priv->p);
	mpz_init(priv->q);

	clock_t keygenTime = std::clock();
	int res = rsac_keygen(pub, priv);
	// int res = generate_keys(pub, priv);

	if (res != 0) {
		printf("FAIL: rsac_string_encrypt_decrypt rsac_keygen returned %d, expected 0\n", res);
		fail++;
	}
	printTime(keygenTime);

	printf("\n_________________________Encription_________________________\n\n");

	rsac_encryptV4(pub, message, m_len, c, &c_len);

	printf("\n_________________________Decription_________________________\n\n");

	rsac_decryptV4(priv, *c, c_len, m_result, &result_len);

	printf("\n________________________Final Result________________________\n\n");
	printf("expected:\n'%s' \ngot:\n'%s'\n", message, *m_result);

	free(pub);
	free(priv);
	free(*c);
	free(*m_result);
	if (fail == 0) {
		printf("\nTest PASSED\n");
	}

	return fail;
}

void printGPUProperties() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		cout << "  Number of multiprocesors: " << prop.multiProcessorCount << endl;
		cout << "  Total global memory: " << prop.totalGlobalMem << endl;
		cout << "  Shared memory per block: " << prop.sharedMemPerBlock << endl;
		cout << "  Number of registers per block per block: " << prop.regsPerBlock << endl;
		cout << "  Warp size: " << prop.warpSize << endl;
		cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
	}
	cout << endl;
}

int main() {
	int failures = 0;
	printGPUProperties();
	printf(" CHAR_BIT je: %d\n", CHAR_BIT);
	printf("Velicina char je: %d\n", sizeof(char));
	printf("Velicina unsigned char je: %d\n", sizeof(unsigned char));
	printf("Velicina mp_limb_t je: %d\n", sizeof(mp_limb_t));
	printf("Velicina unsigned long int je: %d\n", sizeof(unsigned long int));
	printf("Velicina unsigned long long int je: %d\n", sizeof(unsigned long long int));

	failures += test_rsac_string_encrypt_decrypt4();

	printf("%d failures\n", failures);
	return failures > 0;
}