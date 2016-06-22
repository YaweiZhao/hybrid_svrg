#include <iostream>
#include <fstream>
#include <cstring>
#include "file.h"
using namespace std;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        bool file::write(string text,int mode)
	{
	    bool success=false;
            ofstream outfile;
            outfile.open(fn.c_str(),mode);
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
