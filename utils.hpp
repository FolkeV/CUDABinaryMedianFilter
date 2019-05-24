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

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "medianFilterParam.hpp"

namespace util{

	void getPadding(size_t numCols, size_t numRows, size_t* nCols, size_t* nRows){
		// Calculate padding parameters
		size_t row_nCols = numCols % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 ?
					numCols : numCols + (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) - numCols %(ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
		size_t row_nRows = numRows %  ROWS_BLOCKDIM_Y == 0 ?
				numRows : numRows +  ROWS_BLOCKDIM_Y - numRows % ROWS_BLOCKDIM_Y;

		size_t col_nCols = numCols % COLUMNS_BLOCKDIM_X == 0 ?
				numCols : numCols +  COLUMNS_BLOCKDIM_X - numCols%COLUMNS_BLOCKDIM_X;
		size_t col_nRows = numRows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 ?
					numRows : numRows + (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) - numRows % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
		*nCols = std::max(row_nCols, col_nCols);
		*nRows = std::max(row_nRows, col_nRows);
	}

	int mean(const unsigned char* img, const int N){
		int mean = 0;

		for(int i = 0; i<N; i++){
			mean += img[i];
		}
		mean /= N;

		return mean;
	}

	void threshold(unsigned char* outputImg, unsigned char* inputImg, size_t mean, size_t N){
		for (int i = 0; i < N; i++){
			outputImg[i] = inputImg[i] < mean ? 255:0;
		}
	}

	cv::Mat postProc(unsigned int* img, size_t numCols, size_t numRows){
		// Initilise a Mat to the correct size and all zeros
		cv::Mat outputImg(numRows, numCols, CV_8UC1, cv::Scalar::all(0));

		for (int i = 0; i < numRows; i++){
			for(int j = 0; j< numCols; j++){
				size_t idx = i * numCols + j;
				if (img[idx] > 0){
					// The background will 0, force all labels a bit away from zero
					// to be able to visualise better
					outputImg.at<uchar>(i,j) = img[idx]%240+15;
				}
			}
		}

		applyColorMap(outputImg, outputImg, cv::COLORMAP_JET);

		return outputImg;
	}
}




#endif /* UTILS_HPP_ */
