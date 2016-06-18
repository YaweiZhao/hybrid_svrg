#pragma once

/*!
* file util.h
* \brief Struct Option stores many general arguments in model
*/

#include <cstring>
#include <cstdlib>
#include "constant.h"

//#include "AzsSvrg.hpp"

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        struct Option
        {
            /*---  parameters  ---*/
            bool do_dense, do_sparse; 
            char* fn_trn_x;
            char* fn_tst_x;
            char* fn_trn_y;
            char*  fn_tst_y;

            bool is_pipeline;
            int thread_cnt;
            int epoch_size; 
            int num_servers;
            int num_aggregator;
            int lock_option;
            int num_lock;
            int max_delay;
            int data_block_size;
            int max_preload_data_size;
            
            int dimention;
            int data_size;
            double regularized;
            double learning_rate;
            double gamma;
            int max_num_iteration;
            bool class_num;            

            Option();
            /*!
            * \brief Get the model-set arguments from file  
            */
            void parseArgs(const char *inp_param);
            void printParam() const;
            void setParam(const char* fn);           
        };

        std::string GetSystemTime();
        extern std::string g_log_suffix;
    }
}
