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

void addTwoBigNumbers(mpz_t result, mpz_t a, mpz_t b) {
	mpz_add(result, a, b);
}