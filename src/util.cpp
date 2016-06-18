#include<iostream>
#include "util.h"
#include <cstring>
#include <unordered_map>
#include <sstream>
#include <fstream>
using namespace std;
namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        Option::Option()
        {
            do_dense = false;
            do_sparse = false; 

            is_pipeline = false;
            thread_cnt =1;
            epoch_size = 1;
            data_block_size = 2000000000;
            max_preload_data_size = 2000000000;

            //multiverso config
            num_servers = 0;
            num_aggregator = 1; 
            lock_option = 1;
            num_lock = 100;
            max_delay = 0;
            class_num=0;
    
        }  

        void Option::parseArgs(const char *inp_param)
        {
            const char *param = inp_param; 
            if (*param == '@') {
                /*---  read paramaters from a file  ---*/
                const char *fn = param+1; 
                setParam(fn); 
            }
            else
            {
                multiverso::Log::Fatal("The configure file is not provided correctly!\n");
            }
              
        }

        void Option::setParam(const char* fn)
        {
            std::fstream fin(fn);
            std::string buffer;
            if(!fin.is_open()) 
            {
                std::cout<<"Open file error!\n";
                exit(1);
            }
            while(!fin.eof())
            {
                getline(fin,buffer,'\n');
                std::string param_name = "";
                std::string param_value = "";
                std::stringstream item(buffer);
                item>>param_name;
                item>>param_value;
                
                if(param_name=="do_sparse") 
			do_sparse = true;
                if(param_name=="is_pipeline")
			is_pipeline = true;
                if(param_name=="thread_cnt") 
			thread_cnt = atoi(param_value.c_str());
                if(param_name=="num_servers") 
			num_servers = atoi(param_value.c_str());
                if(param_name=="num_aggregator") 
			num_aggregator = atoi(param_value.c_str());
                if(param_name=="lock_option") 
			lock_option = atoi(param_value.c_str());
                if(param_name=="num_lock") 
			num_lock = atoi(param_value.c_str());
                if(param_name=="max_delay") 
			max_delay = atoi(param_value.c_str());
                if(param_name=="data_block_size") 
			data_block_size = atoi(param_value.c_str());
                if(param_name=="max_preload_data_size") 
			max_preload_data_size = atoi(param_value.c_str());
                if(param_name=="dimention") 
			dimention = atoi(param_value.c_str());
                if(param_name=="epoch_size") 
			epoch_size = atoi(param_value.c_str());
                if(param_name=="data_size") 
			data_size = atoi(param_value.c_str());
                if(param_name=="regularized") 
			regularized = atoi(param_value.c_str());
                if(param_name=="learning_rate") 
			learning_rate = atof(param_value.c_str());
                if(param_name=="gamma") 
			gamma = atof(param_value.c_str());
                if(param_name=="class_num") 
			class_num = atof(param_value.c_str());
                if(param_name=="max_num_iteration") 
			max_num_iteration = atof(param_value.c_str());
            }

        }

        std::string GetSystemTime()
        {
            time_t t = time(0);
            char tmp[128];
            strftime(tmp, sizeof(tmp), "%Y%m%d%H%M%S", localtime(&t));
            return std::string(tmp);
        }

        std::string g_log_suffix;
    }
}
