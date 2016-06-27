#include <iostream>
#include <fstream>
#include <cstring>
#include "file.h"
using namespace std;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        file::file(string fn)
	{
	    this->fn = fn;
	}
        bool file::write(string text)
	{
	    bool success=false;
            ofstream outfile;
            outfile.open(fn.c_str(),ios::app);
            if(outfile.is_open())
	    {
		outfile<<text;
		success = true;
            }
	    else 
 	    {
		cout<<"FATAL: Open file fails!\n";
		exit(1);
	    }		
	    outfile.close();
	    return success;
	}

	bool file::read()
        {
	    bool success=false;
	    


	    return success;
        }
    }
}
