# CUDA Binary Median Filter

A median filter for binary images implemented in CUDA. Utilises a separable convolution followed by a threshold to compute the median of each pixel in an area around the pixel. The separable convolution reduces the number of operations needed to be performed and the number of memory reads while the threshold removes the need to sort the neighbourhood list to compute the median in a more traditional implementation of a median filter.

This was one part of a pipeline implemented for a GPU blob detection algorithm during my master's thesis for a degree in Master's of Science in engineering: Engineering Physics.

## Prerequisites

* `OpenCV` is used to load and display images, it is assumed that it has been installed correctly.
* `CUDA-toolkit`, This has been tested on an Nvidia Jetson TX2 running CUDA 9.0. Any newer version of the CUDA toolkit should be usable and many of the older ones as well. It does use managed memory, so your graphics card needs to be compatible with that. Per Nvidia the requirements are:

    * "a GPU with SM architecture 3.0 or higher (Kepler class or newer)"
    * "a 64-bit host application and non-embedded operating system (Linux, Windows, macOS)"

## Compiling

* Clone this repository onto your computer

* Edit the line `CUDAFLAGS = -arch=sm 62` in the makefile to whichever compute capability your graphics card uses. The info should be able to be found here: <https://developer.nvidia.com/cuda-gpus.>

* Run `make`

## Usage

`$ ./<main> <image-file>`

## License

The source code is provided under The MIT license