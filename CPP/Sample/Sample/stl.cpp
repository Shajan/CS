// disable warning C4786: symbol greater than 255 character,
// okay to ignore
#pragma warning(disable: 4786)

#include "stdafx.h"

void FileIO();
void Vector(const vector<int>& vInt);
void Sort(const vector<int>& lines);
void Set();

void GetVector(const vector<string>& lines, vector<int>& vInt);
void PrintVector(const vector<int>& vInt, string name);

void Stl(const vector<string>& lines)
{
//	vector<int> vInt;
//	GetVector(lines, vInt);
//	PrintVector(vInt, "Input");
//	FileIO();
//	Sort(vInt);
//	Vector(vInt);
	Set();
}

void GetVector(const vector<string>& lines, vector<int>& vInt)
{
	vector<string>::const_iterator itStr = lines.begin();

	while (itStr != lines.end())
	{
		string s = *itStr;
		int i = atoi(s.c_str());
		vInt.push_back(i);
		++itStr;
	}
}
void FileIO()
{
#define INPUT_FILE "Input.txt"
#define OUTPUT_FILE "Output.txt"

	fstream input(INPUT_FILE, ios::in);
	fstream output(OUTPUT_FILE, ios::out);

	copy(istream_iterator<string>(input), istream_iterator<string>(), ostream_iterator<string>(output, "\n"));
/*
	vector<int> vInt;

	while (!input.eof())
	{
		string oneLine;
		input >> oneLine;
		if ((oneLine.length() != 0) && (oneLine[0] != '#'))
		{
			int iValue = atoi(oneLine.c_str());
			vInt.push_back(iValue);
		}
	}

	input.close();

	sort(vInt.begin(), vInt.end());

	for (vector<int>::const_iterator iter = vInt.begin(); iter != vInt.end(); ++iter)
	{
		output << *iter << endl;
	}
*/
	output.close();
}
void PrintVector(const vector<int>& vInt, string name)
{
	cout << "............[" << name << "]..........." << endl;
/*
	vector<int>::const_iterator itInt = vInt.begin();
	while(itInt != vInt.end())
	{
		cout << *itInt << endl;
		++itInt;
	}
*/
	copy(vInt.begin(), vInt.end(), ostream_iterator<int, char>(cout, "\n"));

	cout << "............[end " << name << "]..........." << endl;
}

void Vector(const vector<int>& vInt)
{
	vector<int> v(25);
	v = vInt;

	cout << "Vector capacity : " << v.capacity() << endl;
	cout << "Vector size : " << v.size() << endl;

	sort(v.begin(), v.end());

}

int IntCompare(const void* p1, const void* p2) { return (*(int*)p1>*(int*)p2); }
void Sort(const vector<int>& vInt)
{
	vector<int> a;
	vector<int> b;

	PrintVector(vInt, "Input");

	a = vInt;
	sort(a.begin(), a.end());
	PrintVector(a, "sort()");

	b = vInt;
	qsort(b.data(), b.size(), sizeof(int), IntCompare);
	PrintVector(a, "qsort()");
}
 
void Set()
{
	const char* a[6] = {"a1", "a2", "a3", 
                      "a&b1", "a&b2", "a&b3"};
	const char* b[6] = {"a&b3", "a&b2", "a&b1",
                      "b3", "b2", "b1"};

  set<const char*> A(a, a + 6);
  set<const char*> B(b, b + 6);
  set<const char*> C;

  cout << "Set A: ";
  copy(A.begin(), A.end(), ostream_iterator<const char*>(cout, " "));
  cout << endl;
  cout << "Set B: ";
  copy(B.begin(), B.end(), ostream_iterator<const char*>(cout, " "));   
  cout << endl;

  cout << "Union: ";
  set_union(A.begin(), A.end(), B.begin(), B.end(),
            ostream_iterator<const char*>(cout, " "));   
  cout << endl;

  cout << "Intersection: ";
  set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                   ostream_iterator<const char*>(cout, " "));    
  cout << endl;

  cout << "Difference: ";
  set_difference(A.begin(), A.end(), B.begin(), B.end(),
                   ostream_iterator<const char*>(cout, " "));    
  cout << endl;

  set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(),
                 inserter(C, C.begin()));

  cout << "Set C (Symmetrical difference of A and B): ";
  copy(C.begin(), C.end(), ostream_iterator<const char*>(cout, " "));
  cout << endl;
}
