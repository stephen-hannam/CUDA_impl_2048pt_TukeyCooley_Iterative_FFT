#include <stdio.h>
#include <math.h>

#ifndef gpuAssert
#define gpuAssert( condition ) { if( (condition) != cudaSuccess ) { fprintf( stderr, "\n FAILURE %s in %s, line %d\n", cudaGetErrorString(condition), __FILE__, __LINE__ ); exit( 1 ); } }
#endif

#define PREC 8

// I've used a snippet of someone elses code, and one condition of use is
// to retain this licensing message
// I obtained the code from: 
// https://github.com/parallel-forall/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu

/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/******************************************************************/
/* borrowed just one function for AtomicMax on floats from github */
	 
/**
* @brief Compute the maximum of 2 single-precision floating point values using an atomic operation
*
* @param[in]	address	The address of the reference value which might get updated with the maximum
* @param[in]	value	The value that is compared to the reference in order to determine the maximum
*/
static __device__ void AtomicMax(float * const address, float value)
{
	if (*address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = *address_as_i, assumed;

	do
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}
/******************************************************************/

// don't actually need this, as you can & with offsets instead
//#define B01_MASK 0b0000000000000001 
//#define B02_MASK 0b0000000000000011
//#define B03_MASK 0b0000000000000111
//#define B04_MASK 0b0000000000001111
//#define B05_MASK 0b0000000000011111
//#define B06_MASK 0b0000000000111111
//#define B07_MASK 0b0000000001111111
//#define B08_MASK 0b0000000011111111
//#define B09_MASK 0b0000000111111111
//#define B10_MASK 0b0000001111111111

// * bitwise div of power 2 num (i\n): i >> log2(n)
// * bitwise modulo of power of 2 num(i % n) : i & (n - 1)

// much of the efficiency of this implementation hinges of being able to apply 
// very fast bitwise manipulations which will only be reliable in cases where
// the fundamental dimensionality of the data is strictly by powers of 2
// happily this is the case in this application, and for fft in general

// more specifically, exploiting the luxury of powers of 2 enables the construction
// of blocks that should be (theoretically) entirely divergence free, which also means
// that the __syncthreads statements, though necessary to ensure correctness, are in practice
// often a "formality", as all warps should be finishing at the same time, provided all
// memory accesses to shared and constant memory can be performed in the same number of 
// clock cycles, this is also one of the reasons behind using constant and shared memory.

// though this implementation fixes on 2048 point fft, it could easily be generalised
// to any 2^s point fft needed. There is a happy coincidence, where my particular GPU
// is limited to 1024 threads per block, and a 2048 point iterative fft unrolls its
// two inner loops to 1024 every time. This is optimal, but only by coincidence. Were
// this happy coincidence not to be relied upon, then further optimisations which take
// intra-block synchronisation, and possibly where warp boundaries will cross, might need 
// to be included.

// precision: it must also be mentioned that a loss of precision is being spent on performance
// Nvidia GPUs with Compute Cabaility 5.0 can only assure atomic arithmetic for single precision
// floats, and the registers available to the threads are also 32 bit registers. For this reason,
// the decision was made to reduce the precision from double to single

#define TWIDDLES_LEN 2047
#define LGSAMP 11
#define WSAMP 2048
#define HWSAMP (WSAMP / 2)
#define IDX_SHIFT (32 - LGSAMP)

//typedef float3 Complex; // padding to avoid bank conflicts - might not be needed in CUDA 9.0
typedef float2 Complex;
static __device__  inline Complex ComplexCast(float a, float b);
static __device__  inline Complex ComplexAdd(Complex a, Complex b);
static __device__  inline Complex ComplexSub(Complex a, Complex b);
static __device__  inline Complex ComplexMul(Complex a, Complex b);
static __device__ inline float ComplexNorm(Complex a);

static __device__  inline Complex ComplexCast(float a, float b)
{
	Complex c;
	c.x = a; c.y = b; 
	//c.z = 0;
	return c;
}

// Complex addition
static __device__ inline Complex ComplexAdd(Complex a, Complex b)
{
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	//c.z = 0; // padding to avoid bank conflicts - might not be needed in CUDA 9.0
	return c;
}

// Complex subtraction
static __device__ inline Complex ComplexSub(Complex a, Complex b)
{
	Complex c;
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	//c.z = 0; // padding to avoid bank conflicts - might not be needed in CUDA 9.0
	return c;
}

// Complex multiplication
static __device__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	//c.z = 0; // padding to avoid bank conflicts - might not be needed in CUDA 9.0
	return c;
}

// Fast calc of Norm of a Complex Number
static __device__ inline float ComplexNorm(Complex a) { return sqrt(a.x*a.x + a.y*a.y); }

__device__ __constant__ int offsets[LGSAMP];
__device__ __constant__ float twiddles_re[TWIDDLES_LEN];
__device__ __constant__ float twiddles_im[TWIDDLES_LEN];

__device__ float * in_re;
__device__ float * in_im;

static __device__ float maxFFT = 0;

// would be really good to use this with Pinned Memory
__global__ void core(float * out, int size, int gridNum, int gridWidth)
{
	extern __shared__ Complex signalChunk[]; // twice the size of 1024 threads in block

	int tidx0, tidx1, rtidx0, rtidx1, stridx0, stridx1, idx_twid; 
	Complex radius;
	int start = (gridNum * gridWidth + blockIdx.x) * blockDim.x; // start of the block, threadIdx.x == 0
	
	tidx0 = threadIdx.x; 
	tidx1 = threadIdx.x + HWSAMP;
	rtidx0 = __brev(tidx0) >> IDX_SHIFT; 
	rtidx1 = __brev(tidx1) >> IDX_SHIFT;
	stridx0 = rtidx0 + start; 
	stridx1 = rtidx1 + start;
	
	if(tidx0 + start < size){
					
		// permutation by bit reversal of array indices as 11 bit numbers, requires 32-11 bit shift
		signalChunk[tidx0] = ComplexCast(in_re[stridx0], in_im[stridx0]);
		signalChunk[tidx1] = ComplexCast(in_re[stridx1], in_im[stridx1]);
		__syncthreads();
		
		for (unsigned int s = 0; s < LGSAMP; s++) // 11 values of s as per iterative algorithm for fft (2048 point)
		{
			tidx0 = threadIdx.x + (threadIdx.x & (~offsets[s]));
			tidx1 = tidx0 + offsets[s] + 1;
			idx_twid = offsets[s] + (threadIdx.x & offsets[s]);
			
			radius = ComplexMul(signalChunk[tidx1], ComplexCast(twiddles_re[idx_twid], twiddles_im[idx_twid])); // A[j + k + m/2]*w
			signalChunk[tidx0] = ComplexAdd(signalChunk[tidx0], radius); // A[j + k]       = A[j + k] + A[j + k + m/2]*w
			signalChunk[tidx1] = ComplexSub(signalChunk[tidx0], radius); // A[j + k + m/2] = A[j + k] - A[j + k + m/2]*w
			__syncthreads();
		}
		tidx0 = threadIdx.x; 
		out[tidx0 + start] = ComplexNorm(signalChunk[tidx0]);
		
		// eval for maxFFT, use AtomicMax() here
		AtomicMax(&maxFFT, out[tidx0 + start]);
	}
}
// use kernel termination as sync point to launch this next kernel named normaliseFinal
__global__ void normaliseFinal(float * out, int size, int gridNum, int gridWidth)
{

	int tidx =  (gridNum * gridWidth + blockIdx.x) * blockDim.x + threadIdx.x;
	if(tidx < size){
		out[tidx] = out[tidx] / maxFFT;
	}
}

// would be really good to use this with Pinned Memory
// since CPU will be busy reading in wavefile while this is happening
// this doesn't need to be all that optimised, it could even be as sequential
// as the CPU version
// NOTE: this should be 1 block, 11 threads
__global__ void calcTwiddles(float * out_re, float * out_im, float * zerothTwiddles_re, float * zerothTwiddles_im)
{
	int tid = threadIdx.x;
	Complex runVal; runVal.x = 1; runVal.y = 0; //runVal.z = 0;
	
	for (int i = offsets[tid]; i < 2 * offsets[tid] + 1; i++)
	{
		out_re[i] = runVal.x;
		out_im[i] = runVal.y;
		runVal = ComplexMul(runVal, ComplexCast(zerothTwiddles_re[tid], zerothTwiddles_im[tid]));
	}
}

void writeBinFile(const char * file_name, float * bin_data, size_t nmemb){
	FILE * fp = fopen(file_name, "wb");
	printf("writing %s is writing %d in %lu bytes\n",file_name, nmemb, sizeof(float) * nmemb);
	fflush(stdout);
	fwrite(bin_data, sizeof(float), nmemb, fp);
	fflush(fp);
	fclose(fp);
	
}

size_t readBinFile(const char * file_name, float ** bin_data){
	FILE * fp = fopen(file_name,"r");
	fseek(fp, 0, SEEK_END);
	size_t sz = ftell(fp);
	size_t nmemb = sz/sizeof(**bin_data);
	printf("file %s is %lu elements in %lu bytes\n", file_name, nmemb, sz);
	fflush(stdout);
	rewind(fp);
	*bin_data = (float*)malloc(sz);
	fread(*bin_data, sizeof(float), nmemb, fp);
	fflush(fp);
	fclose(fp);
	return nmemb;
}

int main(int argc, char ** argv){
	printf("hello\n");
	fflush(stdout);
	const float pi = 3.1415927410;
	const char * wave_bin = "wave_out";
	const char * timefreq_ref = "timefreq_out";
	const char * timefreq_cmp = "timefreq_done";
	const int h_offsets[LGSAMP] = {0,1,3,7,15,31,63,127,255,511,1023};
	
	for(int i = 0; i < LGSAMP; i++) printf("h_offsets[%d] = %d, ", i, h_offsets[i]);
	printf("\n");
	fflush(stdout);
	
	float h_zerothTwiddles_re[LGSAMP];
	float h_zerothTwiddles_im[LGSAMP];
	
	float h_twiddles_re[TWIDDLES_LEN];
	float h_twiddles_im[TWIDDLES_LEN];
	
	float a;
	float * h_wave_data;
	float * h_timefreq_ref_data;
	
	size_t wave_num_els = readBinFile(wave_bin, &h_wave_data);
	
	size_t timefreq_num_els = readBinFile(timefreq_ref, &h_timefreq_ref_data);
	
	// testing bit reversal
	float * out; 
	if((out = (float*)malloc(timefreq_num_els * sizeof(*out))) == NULL)
		printf("failed to allocate out");
	
	float * h_timefreq_cmp_data;
	printf("Allocating h_timefreq_cmp_data\n");
	fflush(stdout);
	if((h_timefreq_cmp_data = (float*)malloc(timefreq_num_els * sizeof(*h_timefreq_cmp_data))) == NULL)
		printf("failed to allocate h_timefreq_cmp_data");
	
	size_t wave_num_pels = ((size_t)ceil((double)wave_num_els / (double)WSAMP)) * WSAMP;
	printf("wave_num_els = %d, wave_num_pels = %d\n", wave_num_els, wave_num_pels);
	fflush(stdout);
	
	float ** h_sig_data;
	printf("Allocating h_sig_data\n");
	fflush(stdout);
	if(h_sig_data = (float**)malloc(2 * sizeof(*h_sig_data))){
		for(int i = 0; i < 2; i++){ 
			printf("Allocating h_sig_data[%d]\n", i);
			if((h_sig_data[i] = (float*)malloc(wave_num_pels * sizeof(**h_sig_data))) == NULL)
				printf("failed to allocate h_sig_data[%d]\n", i);
		}
	}
	else printf("failed to allocate h_sig_data\n");
	
	// cast wave data to complex and add padding
	printf("Casting wave to complex and padding with zeros\n");
	fflush(stdout);
	for(int i = 0; i < wave_num_pels; i++)
	{
		h_sig_data[1][i] = 0;
		if(i < wave_num_els){
			h_sig_data[0][i] = h_wave_data[i];
		}
		else h_sig_data[0][i] = 0;
	}
	
	// calc zerothTwiddles
	printf("Calculating the zeroth twiddles\n");
	fflush(stdout);
	for (int ii = 0; ii < LGSAMP; ii++)
	{
		a = -2 * pi / ((double)(2*(h_offsets[ii] + 1)));
		//printf("2 * pi * ii = %f, (float)h_offsets[ii] + 1 = %f\n", 2 * pi * ii, (float)h_offsets[ii] + 1);
		//fflush(stdout);
		h_zerothTwiddles_re[ii] = (float)cos(a);		
		h_zerothTwiddles_im[ii] = (float)sin(a);
		//printf("a = %f, h_zerothTwiddles_re[%d] = %f, h_zerothTwiddles_im[%d] = %f\n",a,ii,h_zerothTwiddles_re[ii],ii,h_zerothTwiddles_im[ii]);
		//fflush(stdout);
	}
	
	//float chk_twiddles_re[TWIDDLES_LEN];
	//float chk_twiddles_im[TWIDDLES_LEN];
	//
	//for(int s = 0; s < LGSAMP; s++){
	//	float runVal_re = 1.0f; float temp1;
	//	float runVal_im = 0.0f; 
	//	for (int i = h_offsets[s]; i < 2 * h_offsets[s] + 1; i++)
	//	{
	//		chk_twiddles_re[i] = runVal_re;
	//		chk_twiddles_im[i] = runVal_im;
	//		temp1 = runVal_re;
	//		runVal_re = runVal_re * h_zerothTwiddles_re[s] - runVal_im * h_zerothTwiddles_im[s];
	//		runVal_im = temp1 * h_zerothTwiddles_im[s] + runVal_im * h_zerothTwiddles_re[s];
	//	}
	//}
	/********************** kernel stuff **********************/
	
	// wave data in
	printf("Allocating memory on device\n");
	fflush(stdout);
	float * d_sig_data_re; cudaMalloc(&d_sig_data_re, wave_num_pels * sizeof(*d_sig_data_re)); 
	float * d_sig_data_im; cudaMalloc(&d_sig_data_im, wave_num_pels * sizeof(*d_sig_data_im)); 
	// timefreq data out 
	//float * d_timefreq_cmp_data; cudaMalloc(&d_timefreq_cmp_data, timefreq_num_els * sizeof(*d_timefreq_cmp_data)); 
	float * d_out; cudaMalloc(&d_out, timefreq_num_els * sizeof(*d_out)); 
	
	// supporting structures
	float * d_zerothTwiddles_re; cudaMalloc(&d_zerothTwiddles_re, TWIDDLES_LEN * sizeof(*d_zerothTwiddles_re)); 
	float * d_zerothTwiddles_im; cudaMalloc(&d_zerothTwiddles_im, TWIDDLES_LEN * sizeof(*d_zerothTwiddles_im)); 
	float * d_twiddles_re; cudaMalloc(&d_twiddles_re, TWIDDLES_LEN * sizeof(*d_twiddles_re)); 
	float * d_twiddles_im; cudaMalloc(&d_twiddles_im, TWIDDLES_LEN * sizeof(*d_twiddles_im)); 
	//int * d_offsets; cudaMalloc(&d_offsets, LGSAMP * sizeof(*d_offsets));
	
	// copy to device
	printf("Copying data over to device\n");
	fflush(stdout);
	
	gpuAssert( cudaMemcpyToSymbol(offsets, h_offsets, LGSAMP * sizeof(*h_offsets), 0, cudaMemcpyHostToDevice) );
	//cudaMemcpy( d_offsets, h_offsets, LGSAMP * sizeof(*d_offsets), cudaMemcpyHostToDevice );
	gpuAssert( cudaMemcpy( d_zerothTwiddles_re, h_zerothTwiddles_re, LGSAMP * sizeof(*d_zerothTwiddles_re), cudaMemcpyHostToDevice ) );
	gpuAssert( cudaMemcpy( d_zerothTwiddles_im, h_zerothTwiddles_im, LGSAMP * sizeof(*d_zerothTwiddles_im), cudaMemcpyHostToDevice ) );
	
	gpuAssert( cudaMemcpy( d_sig_data_re, h_sig_data[0], wave_num_pels * sizeof(*d_sig_data_re), cudaMemcpyHostToDevice ) );
	gpuAssert( cudaMemcpy( d_sig_data_im, h_sig_data[1], wave_num_pels * sizeof(*d_sig_data_im), cudaMemcpyHostToDevice ) );
	gpuAssert( cudaMemcpyToSymbol(in_re, &d_sig_data_re, sizeof(float *)) );
	gpuAssert( cudaMemcpyToSymbol(in_im, &d_sig_data_im, sizeof(float *)) );
	
	// launch prep kernel
	printf("Launching kernel to calc twiddles\n");
	fflush(stdout);
	calcTwiddles<<<1,11>>>(d_twiddles_re, d_twiddles_im, d_zerothTwiddles_re, d_zerothTwiddles_im);
	cudaDeviceSynchronize();
	
	/* this copy to host then back to device may not be necessary */
	// copy twiddles back to host
	gpuAssert( cudaMemcpy(h_twiddles_re, d_twiddles_re, TWIDDLES_LEN * sizeof(*h_twiddles_re), cudaMemcpyDeviceToHost ) );
	gpuAssert( cudaMemcpy(h_twiddles_im, d_twiddles_im, TWIDDLES_LEN * sizeof(*h_twiddles_im), cudaMemcpyDeviceToHost ) );
	
	// copy twiddles back to device
	gpuAssert( cudaMemcpyToSymbol(twiddles_re, h_twiddles_re, TWIDDLES_LEN * sizeof(*d_twiddles_re), 0, cudaMemcpyHostToDevice) );
	gpuAssert( cudaMemcpyToSymbol(twiddles_im, h_twiddles_im, TWIDDLES_LEN * sizeof(*d_twiddles_im), 0, cudaMemcpyHostToDevice) );
	
	// launch core kernel
	int threadsPerBlock = HWSAMP;
	int blocksPerGrid = wave_num_pels / (2 * threadsPerBlock);
	printf("launching kernel with %d blocks and %d threads per block\n", blocksPerGrid, threadsPerBlock);
	fflush(stdout);
	
	int blockEachLaunch = 32;
	int iter = 0;
	for(; iter < blocksPerGrid/blockEachLaunch; iter++){
		core<<<blockEachLaunch, threadsPerBlock, WSAMP * PREC>>>(d_out, timefreq_num_els, iter, blockEachLaunch);
	}
	if(blocksPerGrid % blockEachLaunch > 0) core<<<blocksPerGrid % blockEachLaunch, threadsPerBlock, WSAMP * PREC>>>(d_out, timefreq_num_els, iter, blockEachLaunch);
	cudaDeviceSynchronize();
	
	/* this copy to host then back to device may not be necessary */
	//// copy results back from device
	//cudaMemcpy(h_timefreq_cmp_data, d_timefreq_cmp_data, timefreq_num_els * sizeof(*h_timefreq_cmp_data), cudaMemcpyDeviceToHost );
	//
	//// copy results back to device
	//cudaMemcpy(d_timefreq_cmp_data, h_timefreq_data, timefreq_num_els * sizeof(*d_timefreq_cmp_data), cudaMemcpyHostToDevice );
	
	// launch kernel to normalise the data
	
	iter = 0;
	for(; iter < blocksPerGrid/blockEachLaunch; iter++){
		normaliseFinal<<<blockEachLaunch, threadsPerBlock>>>(d_out, timefreq_num_els, iter, blockEachLaunch);
	}
	if(blocksPerGrid % blockEachLaunch > 0) normaliseFinal<<<blocksPerGrid % blockEachLaunch, threadsPerBlock>>>(d_out, timefreq_num_els, iter, blockEachLaunch);
	cudaDeviceSynchronize();
	
	
	// copy final results back to host
	//cudaMemcpy(h_timefreq_cmp_data, d_timefreq_cmp_data, timefreq_num_els * sizeof(*h_timefreq_cmp_data), cudaMemcpyDeviceToHost );
	gpuAssert( cudaMemcpy(out, d_out, timefreq_num_els * sizeof(*out), cudaMemcpyDeviceToHost ) );
	
	gpuAssert( cudaFree(d_out) );
	// clean up device memory
	gpuAssert( cudaFree(d_sig_data_re) );
	gpuAssert( cudaFree(d_sig_data_im) );
	gpuAssert( cudaFree(d_zerothTwiddles_re) );
	gpuAssert( cudaFree(d_zerothTwiddles_im) );
	gpuAssert( cudaFree(d_twiddles_re) );
	gpuAssert( cudaFree(d_twiddles_im) );
	
	/********************** /kernel stuff **********************/
	
	//for(int i = 0; i < timefreq_num_els; i++) printf("%f ", out[i]);
	
	//printf("Writing results to timefreq_done. int is %lu bytes.\n", sizeof(int));
	writeBinFile(timefreq_cmp, out, timefreq_num_els);
	
	free(out);
	
	free(h_timefreq_ref_data);
	free(h_timefreq_cmp_data);
	free(h_wave_data);
}

// -gencode arch=compute_50,code=sm_50
