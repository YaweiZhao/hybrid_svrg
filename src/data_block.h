#pragma once

/*!
* \file data_block.h
* \brief Class DataBlock is to store the necessary data for trainer and param_loader
*/

#include "util.h"
#include "multiverso.h"
#include "constant.h"

#include "armadillo"


namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        /*!
        * \brief The class DataBlock stores train for trainer and param_loader
        */
        class DataBlock : public multiverso::DataBlockBase
        {
            public:
                DataBlock(){}
                ~DataBlock();
        
                //void AddSamples(AzDSmat *trn_dsm, AzDSmat *tst_dsm, AzDvect *v_trn_y, AzDvect *v_tst_y);
                void AddSamples(sp_mat trn_x, sp_mat tst_x, vec trn_y, vec tst_y);
	            //void GetSamples(const AzDSmat **trn_dsm, const AzDSmat **tst_dsm, const AzDvect **v_trn_y, const AzDvect **v_tst_y);
                void GetSamples(sp_mat trn_x, sp_mat tst_x, vec trn_y, vec tst_y);

            public:
                sp_mat trn_x, tst_x; 
                vec trn_y, tst_y;

                // No copying allowed
                DataBlock(const DataBlock&);
                void operator=(const DataBlock&);
        };
    }
}
