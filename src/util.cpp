#include "util.h"
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <ifstream>
namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        Option::Option()
        {
            do_no_intercept = false;
            do_dense = false;
            do_sparse = false; 

            is_pipeline = false;
            thread_cnt =1;
            epoch = 1;
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

        /*--------------------------------------------------------*/
        #define kw_do_no_intercept "NoIntercept"
        #define kw_train_x_fn "train_x_fn="
        #define kw_train_tar_fn "train_target_fn="
        #define kw_test_x_fn "test_x_fn="
        #define kw_test_tar_fn "test_target_fn="
        #define kw_do_dense "DenseData"
        #define kw_do_sparse "SparseData"
        #define kw_do_regress "Regression"

        #define kw_thread_cnt "threads="
        #define kw_epoch "epoch="
        #define kw_num_servers "num_servers="
        #define kw_num_aggregator "num_aggregator="
        #define kw_lock_option "lock_option="
        #define kw_num_lock "num_lock="
        #define kw_max_delay "max_delay="
        #define kw_data_block_size "data_block_size="
        #define kw_max_preload_data_size "max_preload_data_size="
        #define kw_init_learning_rate "init_learning_rate="
        #define kw_is_pipeline "pipeline"

        #define kw_dim "dim="
        #define kw_class_num "class_num="
        /*--------------------------------------------------------*/
        void Option::printParam(const AzOut &out) const 
        {
            
        }
        
        void Option::parseArgs(const char *inp_param)
        {
            my_map param_map;
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

        void setParam(char* fn)
        {
            std::ifstream fin(fn, std::ios::in);
            char line[1024]={0};
            std::string param_name = "";
            std::string param_value = "";
            while(fin.getline(line,sizeof(line)))
            {
                std::stringstream item(line);
                line>>param_name;
                line>>param_value;
                switch(param_name)
                {
                    case "do_sparse": do_sparse = true;
                        break;
                    case "is_pipeline": is_pipeline = true;
                        break;
                    case "thread_cnt": thread_cnt ＝ atoi(param_value.c_str());
                        break;
                    case "num_servers": num_servers = atoi(param_value.c_str());
                        break;
                    case "num_aggregator": num_aggregator = atoi(param_value.c_str());
                        break;
                    case "lock_option": lock_option = atoi(param_value.c_str());
                        break;
                    case "num_lock": num_lock = atoi(param_value.c_str());
                        break;
                    case "max_delay": max_delay = atoi(param_value.c_str());
                        break;
                    case "data_block_size": data_block_size = atoi(param_value.c_str());
                                      break;
                    case "max_preload_data_size": max_preload_data_size = atoi(param_value.c_str());
                        break;
                    case "dimention": dim = atoi(param_value.c_str());
                        break;
                    case "epoch_size": epoch_size = atoi(param_value.c_str());
                        break;
                    case "data_size": data_size = atoi(param_value.c_str());
                        break;
                    case "regularized": regularized = atoi(param_value.c_str());
                        break;
                    case "learning_rate": learning_rate = atof(param_value.c_str());
                        break;
                    case "gamma": gamma = atof(param_value.c_str());
                        break;
                    case "class_num": class_num = atof(param_value.c_str());
                        break;
                    case "max_num_iteration": max_num_iteration = atof(param_value.c_str());
                        break;
                    default: multiverso::Log::Fatal("the configure parameter is wrong!\n");


                }
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
