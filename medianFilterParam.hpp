/*
 * medianFilterParam.h
 *
 *  Created on: Apr 30, 2019
 *      Author: Folke Vesterlund
 */

// Parameters needed by multiple files.
// Ugly, but effective solution...

#ifndef MEDIANFILTERPARAM_H_
#define MEDIANFILTERPARAM_H_

// Filter size
#define KERNEL_RADIUS 1 // 1 to 8 are okay with the current block sizes
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

// Block size row convolution
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 4
#define ROWS_HALO_STEPS 1

// Block size column convolution
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define COLUMNS_HALO_STEPS 1

#endif /* MEDIANFILTERPARAM_H_ */
