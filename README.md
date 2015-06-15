# Parallel Asynchronous Block Coordinate Methods


This is a repository for hosting the ARock source codes. We provide support for the
following solvers:

	  - Relaxed Jacobi method for solving linear equations
	  - Operator splitting method for solving regularized regression problems (LASSO, logistic regression)
	  - Operator splitting method for solving classification problems (dual SVM)


## Build

To build the code, simple type:		

``` sh
    cd ARock
    make
``` 

If build is sucessful, you should be able to find the following executable files in the bin folder:

``` sh
    bin/jacobi
    bin/least_square 
    bin/logistic     
```

## Test

  