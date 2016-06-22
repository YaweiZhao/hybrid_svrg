#pragma once

#include "constant.h"
#include "util.h"
#include "armadillo"

using namespace arma;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        class Reader
        {
        public:
            Reader(Option *option): option_(option) {}
            ~Reader(){}
            
            void GetSamples(sp_mat& trn_x, sp_mat& tst_x, vec& trn_y, vec& tst_y);

        public:
            const Option *option_;

            //bool is_dense_matrix(const AzSmat *m) const;

            //No copying allowed
            Reader(const Reader&);
            void operator=(const Reader&);
        };
    }
}
