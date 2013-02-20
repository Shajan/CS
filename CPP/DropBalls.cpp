// Drop two balls from a 100 story building to figure out the breaking height, using minimal number of attempts
#include <iostream>
#include <vector>

using namespace std;

bool fDebug=false;

int main(int argc, char* argv[])
{
  int i=100;
	int delta = 3;
	int count=0;

	while (i>=0) {
		if (i != 100) cout << ", ";
		cout << i;
		i -= delta;
		++delta;
		++ count;
	}
	cout << endl << count << endl;
	cin >> i;
    return 0;
}
