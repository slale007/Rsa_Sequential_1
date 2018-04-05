#include "stdafx.h"
#include "timeutil.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <string.h>

using namespace std;

void printTime(clock_t start) {
	cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}



