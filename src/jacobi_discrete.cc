#include <iostream>
#include <omp.h>  
#include "algebra.h"
#include "matrices.h"
#include "jacobi.h"
#include "time.h"

#define max(a, b)  (((a) > (b)) ? (a) : (b)) 

// calculate the residual ||Ax-b||_2
template <typename T>
double calculate_residual(T& A, Vector& b, Vector& x)
{
  Vector Ax(A.rows(), 0.); /*generating a vector at every step may be expensive*/
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

/*Proposed Changes: Brent Edmunds June 2015*/
/*minimum computational time per iteration (speed per thread, Deterministic) */
/*randomized wait times per iteration- simulate a busy network, Robert's stuff, heuristics on stepsize */
/*precompute wait times, then make them available through shared directive, (taint the cache, 1 cache miss) */
/*speed of (computation per iteration per thread local variable, difficulty is that relaunching threads will shuffle who gets what*/
/*manual assignment of cores? timings do not maintain validity between runs*/
/*c++11 introduced lambda functions, which allow us to make our framework a lot more user friendly */                                                                 

template <typename T>
void jacobi(T& A, Vector& b, Vector& x, Parameters& para,double min_comp_time, bool& done) 
{
  /* Brent Edmunds: Currently Implementing: minimum computational time per iteration*/
  int local_m      = 0;
  int local_start  = 0;
  int local_end    = 0;
  int MAX_ITER     = para.MAX_EPOCH;
  bool flag        = para.flag;
  int thread_count = omp_get_num_threads();
  int my_rank      = omp_get_thread_num();
  //double STEP_SIZE = min_comp_time/*./((double)(thread_count))*/;
  double STEP_SIZE = 1.0;
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
  double execution_time=0;
  double time_start=omp_get_wtime();
  int iterations=0;
  for(int itr=0;itr<MAX_ITER; itr++)
  {
    
    for(i=local_start;i<local_end;i++)
    {
      /* bracketing computational work with omp_get_wtime() to calculate computational time */
     
      /*omp does not have a general sleep function that is useful */
      
      /*
	timespec t; //nano_clock method
	clock_gettime(CLOCK_REALTIME,&t); //nano_clock method
	timespec tt=t;
      */
      if(!done){
	iterations++;
      /*omp_get_wtime() is in seconds */
        double start=omp_get_wtime();
	
	  idx = i;
	  Ax_i = dot(A, x, idx);
	  x_i = x[idx];
	  tmp = (b[idx] - Ax_i + A(idx, idx) * x_i) / A(idx, idx) - x_i;

	  //delay(max(min_comp_time-(omp_get_wtime()-start),0));
	  //delay(min_comp_time-(omp_get_wtime()-start));
	  while( (omp_get_wtime()-start) < (min_comp_time)){}	
	  x[idx] += STEP_SIZE * tmp;

	  execution_time+=(omp_get_wtime()-start);
	  /*
	  //clock_nanosleep method
	  t.tv_nsec+=(min_comp_time*1000000000);
	  if(t.tv_nsec>999999999){
	  t.tv_sec+=1;t.tv_nsec-=1000000000; 
	  }
	  
	  while(clock_nanosleep(CLOCK_REALTIME,TIMER_ABSTIME,&t,NULL)!=0){
	  std::cout << "waiting";
	  }
	  timespec s;
	  clock_gettime(CLOCK_REALTIME,&s);
	  long nsecdelta=s.tv_nsec-tt.tv_nsec;
	  time_t secdelta=s.tv_sec-tt.tv_sec;
	  double temp=secdelta+nsecdelta/1000000000.0;
	  execution_time+= temp;//(omp_get_wtime()-start);
	  */
      }
	  
	  
    }
    
    if(flag)
    {
      #pragma omp barrier
      if(my_rank==0 && itr%10==0)
        cout<< calculate_residual(A, b, x)<<endl;
      #pragma omp barrier
    }
   
  }
  done=true;
  double time_end=omp_get_wtime();
  std::cout << my_rank << ":min_comp_time " << min_comp_time << ", average comp time: " << (execution_time)/(iterations) << std::endl;

  
  return;
}


// include this to avoid compilation error for sparse matrix
template void jacobi<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&, Vector&, Vector&, Parameters&,double min_comp_time,bool& done);

// include this to avoid compilation error for dense matrix
template void jacobi<Matrix >(Matrix&, Vector&, Vector&, Parameters&,double min_comp_time,bool& done);
