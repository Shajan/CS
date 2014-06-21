#include "stdafx.h"

class A
{
public:
	A() { m_int = ++s_int; cout << "A::A() " << m_int << endl; }
//	A(const A& a) { m_int = a.m_int; cout << "A::A(A& " << m_int << ")" << endl; }
//	A& operator = (const A& a) { m_int = a.m_int; cout << "A::= " << m_int << endl; return *this; }
	void Print() { cout << " A[m_int=" << m_int << "]"; }
	
	// Post and pre increment operators
	A& operator ++() { ++m_int; return *this; }
	A operator ++(int) { A temp(*this); ++m_int; return temp; }

	// comparison operators
	bool operator ==(A& a) { return m_int == a.m_int; }
	bool operator !=(A& a) { return m_int != a.m_int; }
	bool operator >(A& a) { return m_int > a.m_int; }
	bool operator <(A& a) { return m_int < a.m_int; }

private:
	int m_int;
	static int s_int;
};
int A::s_int=0;

class B
{
public:
	B() { cout << "B::B()" << endl; }
	B(const string s) { m_str = s; cout << "B::(" << s << ")" << endl; }
//	B(const B& b) : m_str(b.m_str), m_a(b.m_a) { cout << "B::(B& " << m_str << ")" << endl; }
//	B& operator = (const B& b) { copy(b); cout << "B::= " << m_str << endl; return *this; }
	void Print() { cout << " B[m_str=" << m_str <<", m_a="; m_a.Print(); cout << "]"; }
private:
	void copy(const B& b) { m_str = b.m_str; m_a = b.m_a; }
	string m_str;
	A m_a;
};

void CopyCtorAssignmentOperator()
{
	//
	// Behavior copy ctor vs. Assignment operator.
	// Use the defined operator else, fallback to member wise copy
	//
	cout << endl << "---- B b1;" << endl;
	B b1;
	cout << "b1 = ";
	b1.Print();

	cout << endl << "---- B b2(\"B\");" << endl;
	B b2("B");
	cout << "b2 = ";
	b2.Print();

	cout << endl << "---- b1 = b2;" << endl;
	b1 = b2;	// assignment operator if defined, else member wise copy
	cout << "b1 = ";
	b1.Print();

	cout << endl << "---- B b3 = b2;" << endl;
	B b3 = b2;	// copy ctor if defined, else member wise copy
	cout << "b3 = ";
	b3.Print();

	cout << endl << "---- B b4(b2);" << endl;
	B b4(b2);	// copy ctor if defined, else member wise copy
	cout << "b4 = ";
	b4.Print();
}

void PreAndPostIncrementOperator()
{
	A a1;
	A a2(a1);
	
	cout << "a1="; a1.Print(); cout << endl;
	cout << "a2="; a2.Print(); cout << endl;

	A aPreIncrement = ++a1;
	A aPostIncrement = a2++;

	cout << "++a1="; aPreIncrement.Print(); cout << endl;
	cout << "a2++="; aPostIncrement.Print(); cout << endl;
}

void ComparisonOperators()
{
	A a1;
	A a2;

	cout << "a1="; a1.Print(); cout << endl;
	cout << "a2="; a2.Print(); cout << endl;

	if (a1 == a2) cout << "a1 == a2"; else cout << "!(a1 == a2)"; cout << endl;
	if (a1 != a2) cout << "a1 != a2"; else cout << "!(a1 != a2)"; cout << endl;
	if (a1 > a2) cout <<"a1 > a2"; else cout << "!(a1 > a2)"; cout << endl;
	if (a1 < a2) cout << "a1 < a2"; else cout << "!(a1 < a2)"; cout << endl;
}
void CLanguage()
{
	//CopyCtorAssignmentOperator();
	//PreAndPostIncrementOperator();
	//ComparisonOperators();
}
