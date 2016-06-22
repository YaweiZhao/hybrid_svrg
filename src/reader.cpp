#include "reader.h"
#include "armadillo"

using namespace std;
using namespace arma;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        void Reader::GetSamples(arma::sp_mat& trn_x, arma::sp_mat& tst_x, arma::vec& trn_y, arma::vec& tst_y) 
        {      
            //const char* fn_trn_x(option_->fn_trn_x);
            //const char* fn_tst_x(option_->fn_tst_x);
            //const char* fn_trn_y(option_->fn_trn_y);
            //const char* fn_tst_y(option_->fn_tst_y);
            multiverso::Log::Debug("training text:%s\n",option_->fn_trn_x.c_str());
            multiverso::Log::Debug("test text:%s\n",option_->fn_tst_x.c_str());

            bool is_load = trn_x.load(option_->fn_trn_x,arma::coord_ascii);
            if(is_load == false) multiverso::Log::Fatal("Load training data from file error!\n");
multiverso::Log::Debug(">>>>>>the training data has been loaded to the temp matrix!\n");
            //set the size of the training data and the dimention of an instance
            trn_x.col(option_->dimention-1).fill(1);
            is_load = tst_x.load(option_->fn_tst_x,arma::coord_ascii);
            if(is_load == false) multiverso::Log::Fatal("Load test data from file error!\n");
multiverso::Log::Debug(">>>>>>the test data has been loaded to the temp matrix!\n");
            is_load = trn_y.load(option_->fn_trn_y);
            if(is_load == false) multiverso::Log::Fatal("Load traing label from file error!\n");
            is_load = tst_y.load(option_->fn_tst_y);
            if(is_load == false) multiverso::Log::Fatal("Load test label from file error!\n");
            multiverso::Log::Debug(">>>>>>the data has been loaded to the temp matrix!\n");
        }
    }
}
