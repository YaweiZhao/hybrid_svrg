#ifndef _FILE_H
#define _FILE_H

#include <iostream>
#include <cstring>
using namespace std;
namespace multiverso
{
    namespace hybrid_logistic_regression
    {
	class file
	{
	    public:
                string fn;
            public:
                file(string ff);
                bool write(string text);

                bool read();
                
        };
        
    }
}




#endif
