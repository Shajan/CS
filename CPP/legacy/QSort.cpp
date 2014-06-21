
#include <iostream>
#include <vector>

using namespace std;

bool fDebug=false;
#define DbgPrint(a) if (fDebug) print(a);
#define DbgPrint1(a, b, c) if (fDebug) print(a, b, c);

void GetRandom(vector<int>& a, int n=10, int max=99);
void print(const vector<int>& a, bool fEndl=true);
void print(const vector<int>& a, int start, int end, bool fEndl=true);

void qsort(vector<int>& a, int start, int end);

int main(int argc, char* argv[])
{
  int tmp;
	vector<int> a;
	GetRandom(a, 100, 1000);
	print(a);
	if (a.size() > 0) {
		qsort(a, 0, a.size()-1);
		for (int i=0; i<a.size()-1; ++i) {
			if (a[i] > a[i+1])
				cout << "Error " << i << ":" << a[i] << ">" << a[i+1] << endl;
		}
	}
	print(a);
	cin >> tmp;
    return 0;
}

void GetRandom(vector<int>& a, int n, int max)
{
	for (int i=0; i<n; ++i)
		a.push_back(rand()%max);
}

void print(const vector<int>& a, bool fEndl)
{
	for (unsigned int i=0; i<a.size(); ++i)
		cout << a[i] << " ";

	if (fEndl)
		cout << endl;
}

void print(const vector<int>&a, int start, int end, bool fEndl)
{
	for (unsigned int i=0; i<a.size(); ++i) {
		if (i != 0) cout << ",";
		if (i == start) cout << "[";
		cout << a[i];
		if (i == end) cout << "]";
	}

	if (fEndl)
		cout << endl;
}

void qsort(vector<int>& a, int start, int end)
{
	if (start >= end)
		return;
	
	DbgPrint1(a, start, end);

	int pivot = a[start];
	int i=start+1, j=end;
	int tmp=0;

	while (i<j) {
		while ((i<=end) && (a[i] < pivot)) ++i;
		while ((j>start) && (a[j] > pivot)) --j;

		if (i<j) {
			tmp = a[i];
			a[i] = a[j];
			a[j] = tmp;
			++i; --j;
			DbgPrint(a);
		}
	}

	if (pivot > a[j]) {
		a[start] = a[j];
		a[j] = pivot;
		DbgPrint(a);
	}

	qsort(a, start, j-1);
	qsort(a, j+1, end);
}
