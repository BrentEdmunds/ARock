/*******************************************************
 * Main function to run the Jacobi code, data is loaded
 * from files.
 *
 * Created: 04/29/2015
 * Update:  06/11/2015 (updated the comments)
 * Author: Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 *******************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <omp.h>  
#include <stdlib.h>
#include "matrices.h"
#include "algebra.h"
#include "jacobi.h" //file doesn't exist
#include "MarketIO.h"

// display help message when the input is not recognized
void exit_with_help();

// parse the input from the terminal
void parse_input_argv(Parameters&, int, char**, std::string&, std::string&, int& );

// main function
int main(int argc, char *argv[])
{
  bool flag = false; // initialize the output flag to false
  int n     = 0;     // this is the size of the linear equation
  
  std::string data_file_name;  // file name for the data matrix, get from the input
  std::string label_file_name; // file name for the data vector, get from the input
  Parameters para;             // parameters
  int total_num_threads = 1;   // total number of threads
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, total_num_threads);
  
  // a vector of vectors to hold results
  vector< vector<double> > result(total_num_threads, vector<double>(2));
  
  vector<double> minimum_comp_time(total_num_threads,0); /* this should become a param later*/
  for(size_t i=0;i<total_num_threads;i++){
    //random minimum computational time of .01 second to .05 seconds
    minimum_comp_time[i]=(2*i+1)*.0001;

  }
  
  
  // consider the sparse data
  if(para.is_sparse)
  {
    /********************************
     * read the data from the files
     ********************************/
    SpMat  A;
    Vector b;
    loadMarket(A,  data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows(); // set the size of the problem
    
    if(b.size()!=A.cols())
    {
      std::cout<<"The size of A and b don't match!"<<std::endl;
      return 0;
    }
    std::cout<<"% Start ARock for solving linear equation"<<std::endl;
    std::cout<<"---------------------------------------------"<<std::endl;
    std::cout<<"The size of the problem is " <<n<<std::endl;
    
    /*****************************
     * start the parallel solver
     *****************************/
    int thread_count = 1;
    for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
    {
      Vector x(n, 0.); // initialize x to a zero vector
      double start = omp_get_wtime();
      bool done=false; /*stop when one thread is finished */
# pragma omp parallel num_threads(thread_count) shared(A, b, x, para,done)
      {
	
	//start each thread with its own delay
        jacobi(A, b, x, para, minimum_comp_time[ omp_get_thread_num() ],done );
      }
      double end = omp_get_wtime();
      result[thread_count-1][0] = end - start;
      result[thread_count-1][1] = calculate_residual(A, b, x);
    }
  }
  
  // consider the dense matrix data
  if(!para.is_sparse)
  {
    Matrix A;
    Vector b;
    /********************************
     * read data from file
     ********************************/
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    
    n = A.rows(); // set the size for the problem
    if(b.size()!=A.cols())
    {
      std::cout<<"The size of A and b don't match!"<<std::endl;
      return 0;
    }
    std::cout<<"% start parallel ayn to solve linear equation"<<std::endl;
    std::cout<<"---------------------------------------------"<<std::endl;
    std::cout<<"The size of the problem is " <<n<<std::endl;

    /*****************************
     *start the parallel solver
     *****************************/
    int thread_count = 1;
    for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
    {
      Vector x(n, 0.); // initialize x to a zero vector
      double start = omp_get_wtime();
      bool done=false; /*stop when one thread is finished */
# pragma omp parallel num_threads(thread_count) shared(A, b, x, para,done)
      {
	
        jacobi(A, b, x, para , minimum_comp_time[ omp_get_thread_num() ],done );
      }
      double end = omp_get_wtime();
      result[thread_count-1][0] = end - start;
      result[thread_count-1][1] = calculate_residual(A, b, x);
    }
  }
  
  std::cout<<"---------------------------------------------"<<std::endl;
  std::cout<<setw(15)<<"# cores";
  std::cout<<setw(15)<<"time(s)";
  std::cout<<setw(15)<<"||Ax -b||";
  std::cout<<std::endl;
  for(int i=0;i<total_num_threads;i++)
  {
    std::cout<<setw(15)<<setprecision(2)<<i+1;
    std::cout<<setw(15)<<setprecision(2)<<scientific<<result[i][0];
    std::cout<<setw(15)<<setprecision(2)<<result[i][1];
    std::cout<<std::endl;
  }
  std::cout<<"---------------------------------------------"<<std::endl;
  return 0;

}


void exit_with_help()
{
  std::cout<<"The usage for jacobi solver is: \n \
            ./jacobi [options] \n \
              -data       < matrix market file for A >\n \
              -label      < matrix market file for b > \n \
              -is_sparse  < if the data format is sparse or not. default: 1. > \n \
              -nthread    < total number of threads, default: 1. > \n \
              -epoch      < total number of epoch, default: 10. > \n \
              -step_size  < step size, default: 1. > \n \
              -block_size < block size, default: 10. > \n \
              -flag       < flag for output, default: 0. >"<<std::endl;
  abort();
}


void parse_input_argv(Parameters& para,
                      int argc,
                      char *argv[],
                      std::string& data_file_name,
                      std::string& label_file_name,
                      int& total_num_threads)
{
  for (int i = 1; i < argc; ++i)
  {
    if(argv[i][0]!='-') break;
    if(++i>=argc)
      exit_with_help();
    else if(std::string(argv[i-1])== "-is_sparse")
      para.is_sparse = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-epoch")
      para.MAX_EPOCH = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-block_size")
      para.block_size = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-step_size")
      para.step_size = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-data")
      data_file_name = std::string(argv[i]);
    else if(std::string(argv[i-1])== "-label")
      label_file_name = std::string(argv[i]);
    else if(std::string(argv[i-1])== "-nthread")
      total_num_threads = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-flag")
      para.flag = atoi(argv[i]);
    else
      exit_with_help();
  }
  return;
}

