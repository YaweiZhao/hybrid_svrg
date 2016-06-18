#include "memory_manager.h"

namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        MemoryManager::MemoryManager(int block_size)
        {
            block_size_ = block_size;
        }
        //Request memory for blocks
        void MemoryManager::RequestBlocks(int64 block_number, std::vector<double*>& result)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            for (int64 i = 0; i < block_number; ++i)
            {
                result.push_back(new (std::nothrow) double[block_size_]);
                assert(result[i] != nullptr);
            }
        }
        //Free the memory for blocks
        void MemoryManager::ReturnBlocks(std::vector<double*>& blocks)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            for (size_t i = 0; i < blocks.size(); ++i)
                delete[] blocks[i];
        }

        MemoryManager::~MemoryManager()
        {
			
        }
    }
}
