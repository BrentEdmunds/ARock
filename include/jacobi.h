/****************************************************************************
 *
 * ARock for solving linear equation
 *
 *     Ax = b
 *
 * A is a nxn nonsingular matrix with nonzero diagonal entries.
 *
 * Jacobi iteration method is the following:
 *
 *  x^{k+1} = - D^{-1} R x^k + D^{-1} b
 *         = T(x^k)
 *         
 * where A = D + R. 
 *
 * ARock for linear equations:
 *    1. select i uniformly at random
 *    2. x_i^{k+1} = eta_k/A(i,i) (\sum_j a_ij \hat x_j^k - b_i)
 *
 *
 * Date Created:  01/29/2015
 * Date Modified: 02/17/2015
 *                04/28/2015 (add comments with the Google style, clean the code)
 *                06/10/2015 (check comments)
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 *
 ****************************************************************************/

#ifndef AROCK_INCLUDE_JACOBI_H
#define AROCK_INCLUDE_JACOBI_H
#include "matrices.h"

/***************************************************
 * Calculate the residual, i.e., |Ax - b|_2.
 *
 * Input:
 *    A -- a matrix, can be dense or sparse
 *         (Matrix, SpMat)
 *    b -- a dense vector
 *         (Vector)
 *
 * Output:
 *    result -- the residual
 *              (double)
 *
 ***************************************************/
template <typename T>
double calculate_residual(T& A, Vector& b, Vector& x);


/******************************************************
 *  solve a linear equation Ax = b with ARock
 *
 * Input:
 *      A -- the target data matrix
 *           (Matrix, SpMat)
 *      b -- the vector b
 *           (Vector)
 *      x -- unknowns, initialized to a zero vector
 *           (Vector)
 *   para -- parameters for the algorithm
 *           (struct)
 *      para.MAX_EPOCH  -- the maximum number of epochs,
 *                         default is 100.
 *                        (int)
 *      para.block_size -- the size of block of coordinates
 *                         default is 20.
 *                        (int)
 *      para.step_size  -- the step size, default is 1.
 *                        (double)
 *      para.flag       -- flag for the output, 0 means
 *                        no output, 1 means printing
 *                        residual per 10 epochs.
 *                        (bool)
 * Output -- (none)
 ******************************************************/
template <typename T>
void jacobi(T& A, Vector& b, Vector& x, Parameters& para);

/******************************************************
 * solve a linear equation Ax = b with parallel Jacobi
 * method. The iteration are synchronized after each 
 * epoch.
 *
 * Input:
 *      A -- the target data matrix
 *           (Matrix, SpMat)
 *      b -- the vector b
 *           (Vector)
 *      x -- unknowns, initialized to a zero vector
 *           (Vector)
 *   para -- parameters for the algorithm
 *           (struct)
 *      para.MAX_EPOCH -- the maximum number of epochs
 *                        (int)
 *      para.flag      -- flag for the output, 0 means
 *                        no output, 1 means print the
 *                        residual per 10 epochs.
 *                        (bool)
 * Output -- (none)
 ******************************************************/
template <typename T>
void syn_jacobi(T& A, Vector& b, Vector& x, Parameters& para);


#endif
