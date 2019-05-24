/* MIT License
 *
 * Copyright (c) 2019 - Folke Vesterlund
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Binary median filter. Implemented with a separable convolution followed by a threshold.
 * Separable convolution based on the CUDA sample `convolutionSeparable` by Nvidia.
 */

#include <assert.h>

#include "medianFilter.cuh"
#include "medianFilterParam.hpp"

void medianFilter(unsigned char* outputImg, const unsigned char* inputImg,
		unsigned int* temp, unsigned int* temp2,
		size_t numCols, size_t numRows,
		size_t charPitch, size_t intPitch)
{
	////////////////////////////////////////////////////////////////////////////////
	// Row convolution filter
	////////////////////////////////////////////////////////////////////////////////
	// Round the number of columns and rows up to evenly fill the blocks
	size_t nCols = numCols % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 ?
			numCols : numCols + (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) - numCols %(ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
	size_t nRows = numRows %  ROWS_BLOCKDIM_Y == 0 ?
			numRows : numRows +  ROWS_BLOCKDIM_Y - numRows % ROWS_BLOCKDIM_Y;

	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	assert(nCols % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	assert(nRows % ROWS_BLOCKDIM_Y == 0);

	// Create Grid/Block
	dim3 blocks(nCols / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), nRows / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	convRowKernel<<<blocks, threads>>>(inputImg, temp, numCols, numRows, charPitch/sizeof(char), intPitch/sizeof(int));

	////////////////////////////////////////////////////////////////////////////////
	// Column convolution filter
	////////////////////////////////////////////////////////////////////////////////
	// Round the number of columns and rows up to evenly fill the blocks
	nCols = numCols %  COLUMNS_BLOCKDIM_X == 0 ?
			numCols : numCols +  COLUMNS_BLOCKDIM_X - numCols%COLUMNS_BLOCKDIM_X;
	nRows = numRows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 ?
			numRows : numRows + (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) - numRows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);

	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	assert(nCols % COLUMNS_BLOCKDIM_X == 0);
	assert(nRows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);
	blocks = dim3(nCols / COLUMNS_BLOCKDIM_X, nRows / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
	threads = dim3(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	convColumnKernel<<<blocks, threads>>>(temp, temp2, numCols, numRows, intPitch/sizeof(int));

	////////////////////////////////////////////////////////////////////////////////
	// Threshold --> median
	////////////////////////////////////////////////////////////////////////////////
	const unsigned int limit = 255*((KERNEL_LENGTH*KERNEL_LENGTH)/2+1);
	threshold<<<dim3(8,8), dim3(16,16)>>>(temp2, outputImg, numCols, numRows, limit, intPitch/sizeof(int));
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__
void convRowKernel(const unsigned char* inputImg, unsigned int* outputImg,
		const size_t numCols, const size_t numRows,
		const size_t charPitch, const size_t intPitch)
{
	// Initilise shared memory to speed up later access.
	__shared__ unsigned char s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	 //Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	inputImg  += baseY * charPitch + baseX;
	outputImg += baseY * intPitch  + baseX;

    //Load main data
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = inputImg[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? inputImg[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (numCols - baseX > i * ROWS_BLOCKDIM_X) ? inputImg[i * ROWS_BLOCKDIM_X] : 0;
    }

    // Wait for all threads do this computation as we'll need the result in the next step
    __syncthreads();

    //Compute and store results
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		unsigned int sum = 0;
		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}
		outputImg[i * ROWS_BLOCKDIM_X] = sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convColumnKernel(const unsigned int* inputImg, unsigned int* outputImg,
		const size_t numCols, const size_t numRows,
		const size_t pitch)
{
	__shared__ unsigned int s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	inputImg  += baseY * pitch + baseX;
	outputImg += baseY * pitch + baseX;

	// Load data into shared memory
	//Main data
	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = inputImg[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Upper halo
	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? inputImg[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo
	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (numRows - baseY > i * COLUMNS_BLOCKDIM_Y) ? inputImg[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	// Wait for all threads do this computation as we'll need the result in the next step
	__syncthreads();

    //Compute and store results
	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		unsigned int sum = 0;

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
		}

		outputImg[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
}

// Thresholding function to quickly compute the median of binary data.
// Reads from pitched data and writes into managed (and unpitched) data.
__global__
void threshold (const unsigned int* inputImg,
		unsigned char* outputImg,
		const size_t numCols, const size_t numRows,
		const unsigned int thresholdVal,
		const size_t intPitch)
{
	for (int i = blockIdx.y * blockDim.y + threadIdx.y;
				i < numRows;
				i += blockDim.y * gridDim.y)
	{
		for (int j = blockIdx.x * blockDim.x + threadIdx.x;
				j < numCols;
				j += blockDim.x * gridDim.x)
		{
			size_t idx = i * numCols + j;
			size_t idxInt = i * intPitch + j;
			outputImg[idx] = inputImg[idxInt] >= thresholdVal? 255:0;
		}
	}
}
