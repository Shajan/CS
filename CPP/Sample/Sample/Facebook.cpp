#include "stdafx.h"
using namespace std;

#define DATA_ROOT "C:\\Shajan\\CS\\CPP\\Sample\\Sample\\"
#define INPUT DATA_ROOT "input.txt"
#define DICTIONARY DATA_ROOT "dictionary.txt"

void Hop();

void Facebook()
{
	Hop();
}

void Hop()
{
	unsigned int n;
	ifstream ifs;
	
	ifs.open(INPUT);


	ifs >> n;

	if (!ifs)
	{
		cout << "error reading file " << INPUT << endl;
		return;
	}
	else
	{
		cout << "number read : " << n << endl;
	}

	for (unsigned int i=1; i<=n; ++i)
	{
		if ((i% (3*5)) == 0) cout << "Hop" << endl;
		else if ((i%3) == 0) cout << "Hoppity" << endl;
		else if ((i%5) == 0) cout << "Hophop" << endl;
	}
}