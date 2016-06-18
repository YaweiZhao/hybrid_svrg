#include "distributed_svrg.h"
#include "constant.h"
namespace multiverso
{
    namespace hybrid_logistic_regression
    {
        void Distributed_svrg::Run(int argc, char *argv[])
        {
            g_log_suffix = GetSystemTime();
            option_ = new (std::nothrow)Option();
            assert(option_ != nullptr);
            option_->parseArgs(argv[1]);
            reader_ = new (std::nothrow)Reader(option_);
            this->Train(argc, argv);

            delete option_;
            delete reader_;
        }

        void Distributed_svrg::Train(int argc, char *argv[])
        {
            //The barrier for trainers
            multiverso::Barrier* barrier =
            new multiverso::Barrier(option_->thread_cnt);

            MemoryManager* memory_mamanger = new MemoryManager(option_->dimention);
            logistic_regression* logistic_regression_objs[2] = {new logistic_regression(option_), new logistic_regression(option_)};

            //Step 1, Create Multiverso ParameterLoader and Trainers, 

            //Prepare option_->thread_cnt trainers for multiverso
            std::vector<multiverso::TrainerBase*> trainers;
            for (int i = 0; i < option_->thread_cnt; ++i)
            {
                trainers.push_back(new (std::nothrow)Trainer(i, option_,
                    barrier, logistic_regression_objs[1], memory_mamanger));
                assert(trainers[i] != nullptr);
            }

            //Prepare ParameterLoader
            ParameterLoader *parameter_loader =new (std::nothrow)ParameterLoader(
                option_, logistic_regression_objs[0]);
            assert(parameter_loader != nullptr);

            //Step 2, prepare the Config for multiverso
            multiverso::Config config;
            config.max_delay = option_->max_delay;
            config.num_servers = option_->num_servers;
            config.num_aggregator = option_->num_aggregator;
            config.is_pipeline = option_->is_pipeline;
            config.lock_option = static_cast<multiverso::LockOption>(option_->lock_option);
            config.num_lock = option_->num_lock;

            //Step3, Init the environment of multiverso
            multiverso::Multiverso::Init(trainers, parameter_loader, config, &argc, &argv);

            char log_name[100];
            sprintf(log_name, "log_%d.txt", /*g_log_suffix.c_str()*/ multiverso::Multiverso::ProcessRank());
            multiverso::Log::ResetLogFile(log_name);

            //Mark the node machine number
            process_id_ = multiverso::Multiverso::ProcessRank();

            //Step 4, prepare the sever/aggregator/cache Table for parametertable(3 or 5) 
            //and initialize the severtable for inputvector
            PrepareMultiversoParameterTables();
            
            //Step 5, start the Train of NN
            TrainModel();

            delete barrier;
            delete memory_mamanger;
            delete logistic_regression_objs[0];
            delete logistic_regression_objs[1];
            for (auto trainer : trainers)
            {
                delete trainer;
            }
            delete parameter_loader;
            multiverso::Multiverso::Close();
        }

        void Distributed_svrg::PrepareMultiversoParameterTables()
        {
            multiverso::Multiverso::BeginConfig();
            int proc_count = multiverso::Multiverso::TotalProcessCount();

            //Create tables, the order of creating tables should arise from 0 continuously
            //The elements of talbes will be initialized with 0
            CreateMultiversoParameterTable(kWeightTableId,
                option_->class_num, option_->dimention,
                multiverso::Type::Double, multiverso::Format::Sparse);

            multiverso::Multiverso::EndConfig();
        }

        //Create the three kinds of tables
        void Distributed_svrg::CreateMultiversoParameterTable(
            multiverso::integer_t table_id, multiverso::integer_t rows,
            multiverso::integer_t cols, multiverso::Type type,
            multiverso::Format default_format)
        {
            multiverso::Multiverso::AddServerTable(table_id, rows,
                cols, type, default_format);
            multiverso::Multiverso::AddCacheTable(table_id, rows,
                cols, type, default_format, 0);
            multiverso::Multiverso::AddAggregatorTable(table_id, rows,
                cols, type, default_format, 0);
        }

        //Get the size of filename, it should deal with large files
        int64 Distributed_svrg::GetFileSize(const char *filename)
        {
#ifdef _MSC_VER
            struct _stat64 info;
            _stat64(filename, &info);
            return (int64)info.st_size;
#else
            struct  stat info;
            stat(filename, &info);
            return(int64)info.st_size;
#endif  
        }

        void Distributed_svrg::TrainModel()
        {
            std::queue<DataBlock*> datablock_queue;
            int data_block_count = 0;

            //char file_x[200];
            //sprintf(file_x, "%s_%d", option_->s_train_x_fn.c_str(), process_id_); 
            //int64 file_size = GetFileSize(file_x);
            //multiverso::Log::Info("train-file-size:%lld, data_block_size:%lld\n",
            //    file_size, option_->data_block_size);
            start_ = clock();
            multiverso::Multiverso::BeginTrain();
            DataBlock *data_block = new (std::nothrow)DataBlock();
            assert(data_block != nullptr);
            for (int cur_epoch = 0; cur_epoch < option_->epoch_size; ++cur_epoch)
            {
                //reader_->ResetStart();
                multiverso::Multiverso::BeginClock();
                ++data_block_count; 
                if (cur_epoch == 0) {
                       clock_t start = clock();
                        LoadData(data_block, reader_, option_->data_block_size);
                        multiverso::Log::Info("LoadOneDataBlockTime:%lfs\n",
                            (clock() - start) / (double)CLOCKS_PER_SEC);
                }
                PushDataBlock(datablock_queue, data_block);
                multiverso::Multiverso::EndClock();
            }

            multiverso::Log::Info("Rank %d Pushed %d datablocks\n", process_id_, data_block_count);

            multiverso::Multiverso::EndTrain();

            //After EndTrain, all the datablock are done,
            //we remove all the datablocks
            RemoveDoneDataBlock(datablock_queue);
        }
              
        void Distributed_svrg::LoadData(DataBlock *data_block, Reader *reader, int64 size)
        {
            //reader->ResetSize(size);
            sp_mat trn_x, tst_x;
            vec trn_y, tst_y;
            reader->GetSamples(trn_x, tst_x, trn_y, tst_y);
            data_block->AddSamples(trn_x, tst_x, trn_y, tst_y);
        }

        void Distributed_svrg::PushDataBlock(
            std::queue<DataBlock*> &datablock_queue, DataBlock* data_block)
        {
            
            multiverso::Multiverso::PushDataBlock(data_block);
            
            datablock_queue.push(data_block);
        }

        //Remove the datablock which is delt by parameterloader and trainer
        void Distributed_svrg::RemoveDoneDataBlock(
            std::queue<DataBlock*> &datablock_queue)
        {
            while (datablock_queue.empty() == false 
                && datablock_queue.front()->IsDone())
            {
                DataBlock *p_data_block = datablock_queue.front();
                datablock_queue.pop();
                delete p_data_block;
            }
        }

    }
}

