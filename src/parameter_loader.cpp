#include "parameter_loader.h"

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        ParameterLoader::ParameterLoader(Option *option,
            logistic_regression *lr_ptr)
        {
            option_ = option;
            lr_ptr_ = lr_ptr;

            parse_and_request_count_ = 0;
        }
	
        void ParameterLoader::ParseAndRequest(
            multiverso::DataBlockBase *data_block)
        {
            if (parse_and_request_count_ == 0)
            {
                start_ = clock();
            }

            multiverso::Log::Info("Rank %d ParameterLoader begin %d\n",
                multiverso::Multiverso::ProcessRank(), parse_and_request_count_);
            ++parse_and_request_count_;

            // Request the parameter
            RequestTable(kWeightTableId);           

        } 
    }
}
