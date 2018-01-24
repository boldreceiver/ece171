

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>

#define LOOPS 10000
#define USE_ALL_REGS
#define MATRIX_DIMENSION 1024 


////////////////////////////////////////////////////////////////////////////////
//! Simple Matrix Mulitplication Kernal
////////////////////////////////////////////////////////////////////////////////
__global__ void  //Matrix Multiplication Kernal d_A*d_B=d_Y
matrixMulKernel( float* d_A, float* d_B, float*d_Y) 
{
	// global ID of the thread
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	//Compute which element of the result this thread is responsible for    	
	int row = id/MATRIX_DIMENSION;
	int column = id % MATRIX_DIMENSION;
	int result =0;
	for (int i = 0; i<MATRIX_DIMENSION; i++){
	    	result += d_A[row*MATRIX_DIMENSION+i]*d_B[column+i*MATRIX_DIMENSION];
		}
	__syncthreads();
	d_Y[row*MATRIX_DIMENSION+column] = result;
}

int foo() 
{

    cudaSetDevice(1);

    // allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    int array_size = MATRIX_DIMENSION*MATRIX_DIMENSION*sizeof(float);
    checkCudaErrors(cudaMalloc(&d_A, array_size));
    checkCudaErrors(cudaMalloc(&d_B, array_size));
    checkCudaErrors(cudaMalloc(&d_C, array_size));

    // allocate host memory
    float* h_A = (float*) malloc(array_size);
    float* h_B = (float*) malloc(array_size);
    float* h_C = (float*) malloc(array_size);

    //Initialize matrices
    for (int i=0;i<MATRIX_DIMENSION;i++){
    	for (int j=0;j<MATRIX_DIMENSION;j++){
	    int element = i*MATRIX_DIMENSION+j;
    	    h_A[element]=element;
    	    h_A[element]=element/100;
	    }
    }

    //Copy arrays to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, array_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, array_size, cudaMemcpyHostToDevice));
     // execute the kernel
    matrixMulKernel<<< MATRIX_DIMENSION*MATRIX_DIMENSION, 1   >>>( d_A, d_B, d_C);
    
    // copy result from device to host
    checkCudaErrors(cudaMemcpy( h_C, d_C, sizeof(array_size), cudaMemcpyDeviceToHost));

    // cleanup memory
    free( h_A);
    free( h_B);
    free( h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
