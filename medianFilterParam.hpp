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
