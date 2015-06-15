#include <iostream>
#include <omp.h>  
#include "algebra.h"
#include "matrices.h"
#include "jacobi.h"

// calculate the residual ||Ax-b||_2
template <typename T>
double calculate_residual(T& A, Vector& b, Vector& x)
{
  Vector Ax(A.rows(), 0.);
  // calculate Ax = A*x
  multiply(A, x, Ax);
  // Ax = Ax - b
  sub(Ax, b);
  return norm(Ax, 2);
}

// include this to avoid compilation error
template double calculate_residual<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&, Vector&, Vector&);

// include this to avoid compilation error
template double calculate_residual<Matrix>(Matrix&, Vector&, Vector&);


// The new Jacobi method by ARock
template <typename T>
void jacobi(T& A, Vector& b, Vector& x, Parameters& para)
{
  int local_m      = 0;
  int local_start  = 0;
  int local_end    = 0;
  int MAX_ITER     = para.MAX_EPOCH;
  bool flag        = para.flag;
  int thread_count = omp_get_num_threads();
  int my_rank      = omp_get_thread_num();
  //double STEP_SIZE = 1./((double)(thread_count));
  double STEP_SIZE = 1.;
  local_m          = A.rows() / thread_count;
  local_start      = local_m * my_rank;
  local_end        = local_m * (my_rank+1);
  
  int global_n = A.rows();
  // set the ending index to global_n for the last process
  if(my_rank == thread_count - 1) local_end = global_n;
  if(my_rank==0 && flag) cout<<"res_" << thread_count << "= [ ";

  int i = 0, idx = 0;
  double Ax_i;
  double tmp = 0;
  double x_i = 0;
  for(int itr=0;itr<MAX_ITER; itr++)
  {
    
    for(i=local_start;i<local_end;i++)
    {

      idx = i;
      // idx = rand()%global_n;
      Ax_i = dot(A, x, idx);
      x_i = x[idx];
      tmp = (b[idx] - Ax_i + A(idx, idx) * x_i) / A(idx, idx) - x_i;
      x[idx] += STEP_SIZE * tmp;
    }
    if(flag)
    {
      #pragma omp barrier
      if(my_rank==0 && itr%10==0)
        cout<< calculate_residual(A, b, x)<<endl;
      #pragma omp barrier
    }
  }
  if(my_rank == 0 && flag) cout<<"];"<<endl;
  return;
}


/*
template <typename T>
void new_jacobi(T& A, Vector& b, Vector& x, Parameters& para)
{
  int num_thread       = omp_get_num_threads(); // total number of threads
  int my_rank          = omp_get_thread_num();  // thread id
  int local_start      = 0;                     // local starting index
  int local_end        = 0;                     // local pass to end index
  int MAX_EPOCH        = para.MAX_EPOCH;        // max number of epochs
  int global_n         = A.rows();              // number of unknowns
  int block_size       = para.block_size;       // the size of the block of coordinates
  int num_blocks       = global_n/block_size;   // total number of blocks
  int block            = 0;                     // dummy variable for looping with the assigned blocks 
  int block_id         = 0;                     // save the id for randomly picked block
  bool flag            = para.flag;             // initialize the output flag
  double step_size     = para.step_size;        // the step size for the update
  int i                = 0;                     // looping within a block
  int idx              = 0;                     // index for the chosen coordinate
  double Ax_i          = 0.;                    // (Ax)_i
  double tmp           = 0.;                    // a temporary variable
  double x_i           = 0.;                    // x_i
  
  int local_num_blocks = num_blocks/num_thread; // distribute the number of blocks evenly to threads
  int remain_blocks    = num_blocks % num_thread; // the remaining number of blocks
  if(my_rank<remain_blocks) local_num_blocks+=1; // evenly distribute the remaining blocks to the threads
  
  Vector local_dx(global_n - (num_blocks-1)*block_size, 0.); // save the result for a block of dx

  // print out residual as the Matlab style vector
  if(my_rank==0 && flag) std::cout<<"new_jacobi_res_" << num_thread << "= [ ";

  // main loop
  for(int itr=0;itr<MAX_EPOCH; itr++)
  {
    // loop the assigned number of blocks
    for(block=0;block<local_num_blocks;block++)
    {
      // generate a random block id
      block_id = rand()%num_blocks;
      
      // calculate the starting index and ending index for the block
      local_start = block_id*block_size;
      local_end   = (block_id + 1) * block_size;
      if(block_id==num_blocks-1)
        local_end = global_n;

      // loop within a block
      for(i=local_start;i<local_end;i++)
      {
        // choose the index
        idx = i;
        // calculate (Ax)_i
        Ax_i = dot(A, x, idx);
        // copy x[idx] to a local variable
        x_i = x[idx];
        // computation based on the local variable
        tmp = (b[idx] - Ax_i + A(idx, idx) * x_i) / A(idx, idx) - x_i;
        // save the update locally
        local_dx[idx-local_start] = step_size * tmp; 
      }
      
      // write the update to vector in shared memory
      for(i=local_start;i<local_end;i++)
      {
        x[i] += local_dx[i-local_start];
      }  
    }
    // if flag == 1, print the residual
    if(flag==1)
    {
      #pragma omp barrier
      if(my_rank==0 && itr%10==0)
        std::cout<< calculate_residual(A, b, x)<<std::endl;
      #pragma omp barrier
    }
  }
  
  if(my_rank == 0 && flag) std::cout<<"];"<<std::endl;
  return;
}
*/

// include this to avoid compilation error for sparse matrix
template void jacobi<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&, Vector&, Vector&, Parameters&);

// include this to avoid compilation error for dense matrix
template void jacobi<Matrix >(Matrix&, Vector&, Vector&, Parameters&);


// Parallel Synchronous Jacobi method
template <typename T>
void syn_jacobi(T& A, Vector& b, Vector& x, Parameters& para)
{
  int num_thread   = omp_get_num_threads();  // the total number of threads
  int my_rank      = omp_get_thread_num();   // the id for my current thread
  int MAX_EPOCH    = para.MAX_EPOCH;         // maximum number of epochs
  bool flag        = para.flag;              // flag for output type
  double step_size = 1.;                     // the step size
  int global_n     = A.rows();               // number of unknowns
  int i            = 0;                      // index for looping
  int idx          = 0;                      // index for updating
  double Ax_i      = 0.;                     // the ith entry of Ax
  double tmp       = 0.;                     // a dummy variable to hold temporary result
  double x_i       = 0.;                     // the ith entry of x
  int block_size   = A.rows() / num_thread;  // the size of the block
  int local_start  = block_size * my_rank;   // the starting index 
  int local_end    = block_size * (my_rank+1); // pass to end index
  

  // set the ending index to global_n for the last process
  if(my_rank == num_thread - 1) local_end = global_n;
  if(my_rank==0 && flag) std::cout<<"syn_res_" << num_thread << "= [ ";

  Vector S(local_end - local_start, 0.);

  // the main loop
  for(int itr=0;itr<MAX_EPOCH; itr++)
  {
    for(i=local_start;i<local_end;i++)
    {
      // idx = rand()%global_n;
      idx = i;
      Ax_i = dot(A, x, idx);
      x_i = x[idx];
      tmp = (b[idx] - Ax_i + A(idx, idx) * x_i) / A(idx, idx) - x_i;
      S[idx-local_start] = x_i + step_size * tmp;
    }
#pragma omp barrier // set a barrier after computation, make sure each process get the same x
    for (i = local_start; i < local_end; ++i)
    {
      x[i] = S[i-local_start];
    }
#pragma omp barrier // set a barrier after computation, make sure each process get the same x

    if(my_rank == 0 && flag)
    {
      std::cout<< calculate_residual(A, b, x)<<std::endl;
    }
  }
  if(my_rank == 0 && flag) std::cout<<"];"<<std::endl;
  return;
}


// include this to avoid compilation error for sparse matrix
template void syn_jacobi<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&, Vector&, Vector&, Parameters&);

// include this to avoid compilation error for sparse matrix
template void syn_jacobi<Matrix >(Matrix&, Vector&, Vector&, Parameters&);
