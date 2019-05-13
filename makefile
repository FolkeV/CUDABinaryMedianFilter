
##############################################################
NVCC=nvcc
PWD = $(shell pwd)
SRC = ./main.cu
CUDAFLAGS= -arch=sm_62
# LDFLAGS = -L$(PWD) -lOpenNI2
OPT= -g -G
RM=/bin/rm -f

# CUDAFLAGS += -I/usr/include/openni2
# CUDAFLAGS += -I/usr/local/include -I/usr/include
LDFLAGS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

main: main.o  medianFilter.o
	${NVCC} ${CUDAFLAGS} $(LDFLAGS) -o main main.o medianFilter.o

main.o: medianFilter.cuh utils.hpp main.cu
	$(NVCC) $(CUDAFLAGS) -std=c++11 -c main.cu

medianFilter.o: medianFilter.cuh medianFilter.cu medianFilterParam.hpp
	${NVCC} ${CUDAFLAGS} -std=c++11 -c medianFilter.cu

clean:
	${RM} *.o main

