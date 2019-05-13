/*
 * medianFilter.cuh
 *
 *  Created on: Mar 5, 2019
 *      Author: Folke Vesterlund
 */

#ifndef MEDIANFILTER_CUH_
#define MEDIANFILTER_CUH_

void medianFilter(unsigned char* outputImg, const unsigned char* inputImg,
		unsigned int* temp, unsigned int* temp2,
		size_t numCols, size_t numRows,
		size_t charPitch, size_t intPitch);

// CUDA Kernels
__global__
void convRowKernel(const unsigned char* inputImg, unsigned int* outputImg,
		const size_t numCols, const size_t numRows,
		const size_t charPitch, const size_t intPitch);

__global__
void convColumnKernel(const unsigned int* inputImg, unsigned int* outputImg,
		const size_t numCols, const size_t numRows,
		const size_t pitch);

__global__
void threshold(const unsigned int* inputImg, unsigned char* outputImg,
		const size_t numCols, const size_t numRows,
		const unsigned int threshold,
		const size_t intPitch);

#endif /* MEDIANFILTER_CUH_ */
