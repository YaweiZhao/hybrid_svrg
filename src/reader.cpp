#include "reader.h"
#include "armadillo"

using namespace std;
using namespace arma;

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        Reader(Option *option): option_(option) 
        {
            
        }

        void Reader::GetSamples(sp_mat trn_x, sp_mat tst_x, vec trn_y, vec tst_y) 
        {      
            //
            trn_x.load(fn_trn_x,arma::coord_ascii);
            trn_x.row(DIMENTION-1).fill(1);
            //trn_x = x_mat.t();
            tst_x.load(fn_tst_x,arma::coord_ascii);
            trn_y.load(fn_trn_y);
            tst_y.load(fn_tst_y);
        }

        /*--------------------------------------------------------*/
        bool Reader::is_dense_matrix(const AzSmat *m) const
        {
            if (option_->do_dense) return true; 
            if (option_->do_sparse) return false; 
            AzTimeLog::print("Checking data sparsity ... ", log_out); 
            double nz_ratio; 
            m->nonZeroNum(&nz_ratio); 
            AzBytArr s("nonzero ratio = "); s.cn(nz_ratio); 
            AzPrint::writeln(log_out, s); 
            return (nz_ratio >= 0.6666); 
        }
    }
}
