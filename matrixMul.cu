

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

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    //cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{

    cudaSetDevice(1);

    //unsigned int timer = 0;
    // cutCreateTimer( &timer));
    //cutStartTimer( timer);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    


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
    checkCudaErrors(cudaEventRecord(start));
    //Perform Mulitplication on the CPU 
    for (int i = 0; i < MATRIX_DIMENSION; i++){
        for (int j = 0; j < MATRIX_DIMENSION; j++){
            h_C[i*MATRIX_DIMENSION+j] = 0;
            for (int k = 0; k < MATRIX_DIMENSION; k++)
                h_C[i*MATRIX_DIMENSION+j] += h_A[i*MATRIX_DIMENSION+k]*h_B[k*MATRIX_DIMENSION+j];
        }
    }
    checkCudaErrors(cudaEventRecord(stop));
    float milliseconds = -1;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf( "CPU Processing time: %f (ms)\n", milliseconds);


    //Copy arrays to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, array_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, array_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
     // execute the kernel
    matrixMulKernel<<< MATRIX_DIMENSION*MATRIX_DIMENSION, 1   >>>( d_A, d_B, d_C);
    
    // copy result from device to host
    checkCudaErrors(cudaMemcpy( h_C, d_C, sizeof(array_size), cudaMemcpyDeviceToHost));

    //cutStopTimer( timer);
    checkCudaErrors(cudaEventRecord(stop));
    milliseconds = -1;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf( "GPU Processing time: %f (ms)\n", milliseconds);
    // cleanup memory
    free( h_A);
    free( h_B);
    free( h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}
