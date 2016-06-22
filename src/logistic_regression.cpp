#include <iostream>
#include <random>
#include <cmath>
#include <armadillo>
#include <sstream>
#include "logistic_regression.h"
#include "file.h"

using namespace std;
using namespace arma;

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
    parameter = zeros<vec>(DIMENTION);
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
void logistic_regression::train()
{
    //produce the samples by a certain probability distribution
    default_random_engine random(time(NULL));    
    vec sample_epoch = produceSamples(random);

    for(int i=0;i<MAX_NUM_ITERATION;i++)
    {
        //set the epoch size. Defaultly, it is set by the variable: EPOCH_SIZE
        setEpochSize(EPOCH_SIZE);
        //compute the full gradient
        vec full_gradient = computeFullGradient(parameter);
        vec local_parameter = parameter;
        for(int j=0;j<EPOCH_SIZE;j++)
        {
            //compute the reduced variance
            vec vr = computeReducedVariance(parameter, local_parameter, full_gradient, sample_epoch[i*EPOCH_SIZE+j]);
            //update the parameters
            updateParameters(local_parameter, vr, LEARNING_RATE);
        }

        //Identify the parameters for the next iteration
        identifyParameters(local_parameter, parameter);

        //evaluate the loss
        if(i%5==0)
        {
            computeLoss(parameter, training_x, training_y);
        }
    } 
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

//compute the regularized norm's gradient
vec logistic_regression::computeRegularizedGradient(vec& parameter)
{
    vec regularized_gradient = 2*REGULARIZED*parameter;
    return regularized_gradient;
}
//compute the stochastic local gradient
vec logistic_regression::computeStochasticGradient(vec& parameters, int index)
{
    double w_x = as_scalar(parameters.t()*training_x.col(index));
    double y = training_y[index];
    double temp0 = (1/(1+std::exp(y*w_x)))*(-1*y);
    vec gradient(temp0*training_x.col(index));
    return gradient;
}

//compute the full gradient
vec logistic_regression::computeFullGradient(vec& global_parameter)
{
    vec full_gradient = zeros<vec>(DIMENTION);
    for(int i=0;i<DATA_SIZE;i++)//consider the constant
    {
        vec temp0 = computeStochasticGradient(global_parameter, i);
        full_gradient += temp0;
    }
    //add the regularized norm's gradient
    vec temp_regularized = computeRegularizedGradient(global_parameter);
    full_gradient += temp_regularized;
    full_gradient = full_gradient/DATA_SIZE;

    return full_gradient;
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
        loss = loss + -1*log(1/(1+exp(-1*temp*y(i))));
    }
    loss = loss/DATA_SIZE;
    stringstream s_loss;
    s_loss<<loss;
    string loss_str;
    loss_str<<s_loss;
    file f("lr_loss.txt");
    f.write(loss_str,ios::app);
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
void logistic_regression::train_test(int trainer_id)
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
    train();
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
