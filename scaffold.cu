

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


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest();
void foo();
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main() 
{
    runTest();

}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest() 
{

    cudaSetDevice(1);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    float milliseconds = -1;
    checkCudaErrors(cudaEventRecord(start));
    foo();
    checkCudaErrors(cudaEventRecord(stop));
    cudaDeviceSynchronize();
    milliseconds = -1;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    //printf("GPU Processing time: %f (ms)\n", milliseconds);
    printf("%f", milliseconds);
}
