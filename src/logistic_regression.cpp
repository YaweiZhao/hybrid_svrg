#include <iostream>
#include <random>
#include <cmath>
#include <armadillo>
#include <sstream>
#include "logistic_regression.h"
#include "file.h"
//#include "barrier.h"
using namespace std;
using namespace arma;
using namespace multiverso;

//initialize the logistic regression parameter settings
logistic_regression::logistic_regression(){}
logistic_regression::logistic_regression(int data_size,
                            double learning_rate,
                            double regularized,
                            int max_num_iteration,
                            int epoch_size,
                            int dimention,
                            double gamma/*momentum*/)
{
    DATA_SIZE = data_size;
	LEARNING_RATE = learning_rate;
	REGULARIZED = regularized;
	MAX_NUM_ITERATION = max_num_iteration;
	EPOCH_SIZE = epoch_size;
    DIMENTION = dimention;
    parameter = zeros<vec>(DIMENTION);
    full_gradient = zeros<vec>(DIMENTION);
    local_parameter = zeros<vec>(DIMENTION);
    momentum = zeros<vec>(DIMENTION);
    GAMMA = gamma;
}
logistic_regression::logistic_regression(Option* option_)
{
    DATA_SIZE = option_->data_size;
	LEARNING_RATE = option_->learning_rate;
	REGULARIZED = option_->regularized;
	MAX_NUM_ITERATION = option_->max_num_iteration;
	EPOCH_SIZE = option_->epoch_size;
    DIMENTION = option_->dimention;
    sample_epoch = zeros<vec>(EPOCH_SIZE*MAX_NUM_ITERATION);
    parameter = zeros<vec>(DIMENTION);
    full_gradient = zeros<vec>(DIMENTION);
    local_parameter = zeros<vec>(DIMENTION);
    momentum = zeros<vec>(DIMENTION);
    GAMMA = option_->gamma;
    this->option_ = option_;

}


//record the begin time of the training process
void logistic_regression::begin(time_t& begin)
{
    time(&begin);
}
//record the end time of the training process
void logistic_regression::end(time_t& end)
{
    time(&end);
}
//training the parameters. The fl2-regularization is added
void logistic_regression::train(int trainer_id, multiverso::Barrier *barrier)
{
    //produce the samples by a certain probability distribution
    default_random_engine random(time(NULL));    
    if(trainer_id==0) 
    {
        //set the epoch size. Defaultly, it is set by the variable: EPOCH_SIZE
        setEpochSize(EPOCH_SIZE);
        sample_epoch = produceSamples(random);
    }
    barrier->Wait();
    for(int i=0;i<MAX_NUM_ITERATION;i++)
    {
multiverso::Log::Info(">>>>>>>>>>>>>>>>learning thread %dth!\n",trainer_id);
        //compute the full gradient in parallel way
        vec full_gradient_thread = computeFullGradient(parameter,trainer_id, option_->thread_cnt);
        //sum all the full gradient in serial way
        mutex_.lock();
        full_gradient = full_gradient+full_gradient_thread;
        mutex_.unlock();
        local_parameter = parameter;
        barrier->Wait();
        for(int j=trainer_id;j<EPOCH_SIZE;j+=option_->thread_cnt)
        {
            //compute the reduced variance
            vec vr = computeReducedVariance(parameter, local_parameter, full_gradient, sample_epoch(i*EPOCH_SIZE+j));
            //update the parameters
            //add the write lock
            mutex_.lock();
            updateParameters(local_parameter, vr, LEARNING_RATE);
            mutex_.unlock();
        }
        barrier->Wait();
        if(trainer_id == 0)
        {
           //Identify the parameters for the next iteration
           identifyParameters(local_parameter, parameter); 
           //evaluate the loss
           if(i%1==0){
               computeLoss(parameter, training_x, training_y);
	       //parameter.load("parameter.txt");
 	       }
        }//end if
    }//end for 
}

//produce the samples by a certain probability distribution
vec logistic_regression::produceSamples(std::default_random_engine random)
{
    int n_instances = EPOCH_SIZE*MAX_NUM_ITERATION;
    vec samples = zeros<vec>(n_instances);
    uniform_int_distribution<int> dis1(0, DATA_SIZE-1);
    for (int i=0;i<n_instances;i++)
    {
        samples(i) = dis1(random);
    }
    return samples;
}

//set the epoch size. Defaultly, it is set by the variable: EPOCH_SIZE
void logistic_regression::setEpochSize(int size)
{
    EPOCH_SIZE = size;
}

//compute the stochastic local gradient
vec logistic_regression::computeStochasticGradient(vec& parameters, int index)
{
    double w_x = as_scalar(parameters.t()*training_x.col(index));
    double y = training_y[index];
    double temp0 = (1/(1+std::exp(y*w_x)))*(-1*y);
    vec gradient(temp0*training_x.col(index)+2*REGULARIZED*parameters);
    return gradient;
}

//compute the full gradient via multiple threads
vec logistic_regression::computeFullGradient(vec& global_parameter,int trainer_id, int thread_cnt)
{
    vec full_gradient_local = zeros<vec>(DIMENTION);
    for(int i=trainer_id;i<DATA_SIZE;i+=thread_cnt)//consider the constant
    {
        vec temp0 = computeStochasticGradient(global_parameter, i);
        full_gradient_local = full_gradient_local + temp0;
    }
    full_gradient_local = full_gradient_local/DATA_SIZE;

    return full_gradient_local;
}

//compute the reduced variance
vec logistic_regression::computeReducedVariance(vec& global_parameter, vec& local_parameter, vec& full_gradient, int index)
{
    vec vr = zeros<vec>(DIMENTION);
    vec temp0 = computeStochasticGradient(local_parameter, index);
    vec temp1 = computeStochasticGradient(global_parameter, index);
    vr=temp0-temp1+full_gradient;
    return vr;
}


//update the parameters
void logistic_regression::updateParameters(vec& local_parameter, vec& vr, double learning_rate)
{
    vec temp = learning_rate*vr + GAMMA*momentum/*momentum*/;
    local_parameter = local_parameter - temp;
    momentum = temp;
}

//Identify the parameters for the next iteration
void logistic_regression::identifyParameters(vec& local_parameter, vec& global_parameter)
{
    global_parameter = local_parameter;
}

//compute the loss function
double logistic_regression::computeLoss(vec& parameter, sp_mat& x, vec& y)
{
    double loss=0;
    for(int i=0;i<DATA_SIZE;i++)
    {
        double temp = as_scalar(parameter.t()*x.col(i));
        loss = loss - log(1/(1+exp(-1*temp*y(i))));
    }
    loss = loss/DATA_SIZE+REGULARIZED*(as_scalar(parameter.t()*parameter));
    ostringstream s_loss;
    s_loss<<loss<<"\n";
    string loss_str = s_loss.str();
    file f("lr_loss.txt");
    f.write(loss_str);
    //multiverso::Log::Info("The loss now is: %f\n",loss);
    return loss;
}


//get ready to parepare the data from the multiverso framework
void logistic_regression::init(DataBlock* data_block)
{
    sp_mat temp_trn_x;
    sp_mat temp_tst_x;
    vec temp_trn_y;
    vec temp_tst_y;
    
    data_block->GetSamples(temp_trn_x,temp_tst_x,temp_trn_y,temp_tst_y);
    training_x = temp_trn_x.t();
    training_y = temp_trn_y;
}

//the entrance from the multiverso to the training thread, and the training process is finally started.
void logistic_regression::train_test(int trainer_id, multiverso::Barrier *barrier)
{
    int fast, slow, self;
    double wait_time=0;
    multiverso::Multiverso::GetClock(&fast, &slow, &self, &wait_time);
    //log the wait time
    
    //begin training
    multiverso::Log::Info("Process: %d and thread: %d begins!\n", multiverso::Multiverso::ProcessRank(),trainer_id);
    time_t begin_time=0;
    time_t end_time=0;
    begin(begin_time);
    train(trainer_id,barrier);
    end(end_time);
}

//set the parameters when pulling the parameters from multiverso
void logistic_regression::setParameters(std::vector<double*> &blocks)
{
    //NOTICE: since logistic regression conducts the binary classification, the number of class is 2. (Here, class_num=1 means the number of class is 2.)
    for(int i=0;i<option_->class_num;i++)
    {
        for (int j=0;j < option_->dimention;j++)
        {
            parameter[j] = blocks[i][j];//since class_num=1, parameter will be valued once.
        }
    }
}

//get the parameters
void logistic_regression::getParameters(std::vector<double*> &blocks)
{
    for(int i=0;i < option_->class_num; i++)
    {
        for (int j=0;j < option_->dimention;j++)
        {
            blocks[i][j] = parameter[j];
        }
    }
}
