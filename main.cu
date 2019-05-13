/*
 * main.cu
 *
 *  Created on: May 8, 2019
 *      Author: Folke Vesterlund
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "medianFilter.cuh"
#include "utils.hpp"

int main(int argc,char **argv){
	std::string fileName;
	size_t numPixels, numRows, numCols;

	if (argc < 2){
		std::cout << "Usage: "<< argv[0] << " <image file>" << std::endl;
		return(-1);
	}
	fileName = argv[1];

	// Read image
	cv::Mat image;
	image = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data){
		std::cerr << "Couldn't open file" << std::endl;
		return(-1);
	}

	if(!image.isContinuous()){
		std::cerr << "Image is not allocated with continuous data. Exiting..." << std::endl;
		return(-1);
	}
	numCols = image.cols;
	numRows = image.rows;
	numPixels = numRows*numCols;

	// Allocate GPU data
	// Image needs to be padded to remove illegal memory accesses.
	size_t nRows, nCols;
	// Calculate padding size
	util::getPadding(numCols, numRows, &nCols, &nRows);

	size_t charPitch;
	size_t intPitch;
	unsigned char* d_binaryImg;
	unsigned char* d_filteredImg;
	unsigned  int* d_temp;
	unsigned  int* d_temp2;
	cudaMallocPitch(&d_binaryImg  , &charPitch, nCols * sizeof(char), nRows);
	cudaMallocPitch(&d_temp       , &intPitch , nCols * sizeof(int) , nRows);
	cudaMallocPitch(&d_temp2      , &intPitch , nCols * sizeof(int) , nRows);
	// Final image can be stored in managed memory for easier post processing
	cudaMallocManaged(&d_filteredImg, numPixels * sizeof(char));

	// Pre process image
	int imgMean = util::mean(image.data, numPixels);
	util::threshold(image.data, image.data, imgMean, numPixels);

	// Copy image to GPU
	cudaMemcpy2D(d_binaryImg, charPitch, image.data , numCols*sizeof(char), numCols * sizeof(char), numRows, cudaMemcpyHostToDevice);

	// Run kernel
	medianFilter(d_filteredImg, d_binaryImg, d_temp, d_temp2, numCols, numRows,charPitch, intPitch);
	cudaDeviceSynchronize();

	// No need to copy image back from GPU as it is handled by managed memory

	// Plot result
	cv::Mat finalImage(numRows, numCols, CV_8UC1, (void*)d_filteredImg);
	cv::imshow("Labelled image", finalImage);
	cv::waitKey();

	// Free memory
	cudaFree(d_binaryImg);
	cudaFree(d_filteredImg);
	cudaFree(d_temp);
	cudaFree(d_temp2);
}
