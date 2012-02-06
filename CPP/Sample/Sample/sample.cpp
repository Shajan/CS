// Sample.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

void CLanguage();
void Stl(const vector<string>& lines);
void Facebook();
bool OpenFile(const char* pFileName, ifstream& file, vector<string>& lines);

 
int main(int argc, char* argv[])
{
	bool fAll = false;
	bool fCLanguage = false;
	bool fStl = false;
	bool fFacebook = false;
	char* pInputFile = NULL;
	ifstream inputStream;
	vector<string> lines;

	for (int i=0; i<argc; ++i)
	{
		if (_stricmp(argv[i], "-c") == 0)
			fCLanguage = true;
		else if (_stricmp(argv[i], "-stl") == 0)
			fStl = true;
		else if (_stricmp(argv[i], "-facebook") == 0)
			fFacebook = true;
		else if (_stricmp(argv[i], "-i") == 0)
		{
			if (argc > i)
				pInputFile = argv[i + i];
		}
	}
	
	if (pInputFile && !OpenFile(pInputFile, inputStream, lines)) goto Exit;
	if (fAll || fCLanguage) CLanguage();
	if (fAll || fStl) Stl(lines);
	if (fAll || fFacebook) Facebook();

Exit:;
//	char dummy;
//	std::cin >> dummy;
}

bool OpenFile(const char* pFileName, ifstream& file, vector<string>& lines)
{
	file.open(pFileName);

	if (!file.good()) 
	{
			cout << "Error Opening file : " << pFileName << endl;
			return false;
	}

	while (!file.eof())
	{
		string oneLine;
		file >> oneLine;
		if ((oneLine.length() != 0) && (oneLine[0] != '#'))
			lines.push_back(oneLine);
	}
	return true;
}