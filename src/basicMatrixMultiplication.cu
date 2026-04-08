#include <gputk.h>
#include <stdlib.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
	  int row = blockIdx.y * blockDim.y + threadIdx.y;
	  int col = blockIdx.x * blockDim.x + threadIdx.x;

	  if (row < numCRows && col < numCColumns) {
	    float sum = 0.0f;
	    for (int k = 0; k < numAColumns; k++) {
	      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
	    }
	    C[row * numCColumns + col] = sum;
	  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns	;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));


  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((numCColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (numCRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC,
                                        numARows, numAColumns,
                                        numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  gpuTKCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
                        cudaMemcpyDeviceToHost));

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  gpuTKCheck(cudaFree(deviceA));
  gpuTKCheck(cudaFree(deviceB));
  gpuTKCheck(cudaFree(deviceC));

  gpuTKTime_stop(GPU, "Freeing GPU Memory");
  
  gpuTKSolution(args, hostC, numCRows, numCColumns);
  
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
