#ifndef __TIMEUTIL_H__

#include "stdafx.h"
#include <ctime>
#include <iostream>

using namespace std;

void printTime(clock_t start) {
	cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << endl;
}

#endif