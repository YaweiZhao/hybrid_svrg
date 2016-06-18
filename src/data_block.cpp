#include "data_block.h"
#include <armadillo>
using namespace arma;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        DataBlock::~DataBlock()
        {
            //ClearSamples();
        }

        void DataBlock::AddSamples(sp_mat& trn_x, sp_mat& tst_x, vec& trn_y, vec& tst_y)
        {
            this->trn_x = trn_x;
            this->tst_x = tst_x;
            this->trn_y = trn_y;
            this->tst_y = tst_y;
        }

	    //Get the information of the index-th Samples
        void DataBlock::GetSamples(sp_mat &trn_x, sp_mat &tst_x, vec &trn_y, vec &tst_y)
        {
            trn_x = this->trn_x;
            tst_x = this->tst_x;
            trn_y = this->trn_y;
            tst_y = this->tst_y;
        }
    }
}
