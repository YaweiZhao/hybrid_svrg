#include "reader.h"
#include "armadillo"

using namespace std;
using namespace arma;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        void Reader::GetSamples(arma::sp_mat trn_x, arma::sp_mat tst_x, arma::vec trn_y, arma::vec tst_y) 
        {      
            //const char* fn_trn_x(option_->fn_trn_x);
            //const char* fn_tst_x(option_->fn_tst_x);
            //const char* fn_trn_y(option_->fn_trn_y);
            //const char* fn_tst_y(option_->fn_tst_y);
            trn_x.load(option_->fn_trn_x,arma::coord_ascii);
            trn_x.row(option_->dimention-1).fill(1);
            tst_x.load(option_->fn_tst_x,arma::coord_ascii);
            trn_y.load(option_->fn_trn_y);
            tst_y.load(option_->fn_tst_y);
        }
    }
}
