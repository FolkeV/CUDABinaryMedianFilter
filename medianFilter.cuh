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
