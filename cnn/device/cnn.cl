// TODO: Define any constants you'll need
#define ARRAY_DIM 784
#define IMG_DIM 28
#define CONV1_SIZE 5
#define CONV1_INPUT_DIM 1
#define CONV1_FILTER 32
#define CONV1_OUT_SIDE 28
#define CONV1_PAD 32
#define CONV1_PAD_IND (CONV1_SIZE-1)*0.5
#define POOL1_DIM 14
#define CONV2_SIZE 5
#define CONV2_INPUT_DIM 32
#define CONV2_FILTER 64
#define CONV2_OUT_SIDE 14
#define POOL2_DIM 7
#define DENSE1_SIZE 256
#define DENSE1_INPUT_DIM (7*7*64)
#define DENSE2_SIZE 10
#define DENSE2_INPUT_DIM (256*1*1)
#define OPENCL_FLOATP const __attribute__((address_space(16776962))) float *restrict
#define OPENCL_UCHARP const __attribute__((address_space(16776960))) unsigned char *restrict
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

// TODO: If you decide you'd like to write helper functions, you can define them here


/*--------------------MAXPOOL HELPER FUNCTION-------------------*/
/* Function to calculate the largest number of the 4. */
float largest_four(float a, float b, float c, float d)
{
  float ret = a;
  if (b > ret) ret = b;
  if (c > ret) ret = c;
  if (d > ret) ret = d;
  return ret;
}

/* ReLU function. */
float relu(float in)
{
  return in > 0 ? in : 0;
}

/*--------------------CONV1 HELPER FUNCTION-------------------*/
float conv1_input_ind(int filrow, int filcol, int filcnl, int row, int col,
                      OPENCL_UCHARP images, int img_idx)
{
  /* Calculate the offset to index the input array. */
  int offset_row = filrow - CONV1_SIZE/2;
  int offset_col = filcol - CONV1_SIZE/2;
  int indrow = row + offset_row;
  int indcol = col + offset_col;
  /* This is the handle of padding. */
  if (indrow < 0 || indrow > IMG_DIM || indcol < 0 || indcol > IMG_DIM)
    return 0.0;

  float res = 0.0;
  res = images[img_idx + IMG_DIM*CONV1_INPUT_DIM*indrow +
               CONV1_INPUT_DIM*indcol + filcnl];
  return (float)res;
}

float conv1_weight_ind(int filrow, int filcol, int filcnl, int fil,
                       OPENCL_FLOATP conv1_weights)
{
  /* conv1 weights is 5x5x1x32, row col cnl filter. */
  float res = 0.0;
  res = conv1_weights[CONV1_SIZE*CONV1_INPUT_DIM*CONV1_FILTER*filrow +
                      CONV1_INPUT_DIM*CONV1_FILTER*filcol +
                      CONV1_FILTER*filcnl + fil];
  return res;
}

float conv1_convolve(int row, int col, int fil, OPENCL_FLOATP conv1_W,
                     OPENCL_FLOATP conv1_b, OPENCL_UCHARP images, int img_idx)
{
  float res = 0.0;
  /* CONV2 filter has 5x5x1 size, 32 filters*/
  #pragma unroll
  for (int filrow = 0; filrow < CONV1_SIZE; filrow++) {
    #pragma unroll
    for (int filcol = 0; filcol < CONV1_SIZE; filcol++) {
      #pragma unroll
      for (int filcnl = 0; filcnl < CONV1_INPUT_DIM; filcnl++) {
        res += conv1_input_ind(filrow, filcol, filcnl, row, col, images, img_idx) *
               conv1_weight_ind(filrow, filcol, filcnl, fil, conv1_W);
      }
    }
  }
  res += conv1_b[fil];
  res = relu(res);
  return res;
}

/*--------------------CONV2 HELPER FUNCTION-------------------*/

float conv2_input_ind(int filrow, int filcol, int filcnl, int row, int col,
                      float* pool1_out)
{
  /* Calculate the offset to index the input array. */
  int offset_row = filrow - CONV2_SIZE/2;
  int offset_col = filcol - CONV2_SIZE/2;
  int indrow = row + offset_row;
  int indcol = col + offset_col;
  /* This is the handle of padding. */
  if (indrow < 0 || indrow > POOL1_DIM || indcol < 0 || indcol > POOL1_DIM)
    return 0.0;

  float res = 0.0;
  res = pool1_out[POOL1_DIM*CONV2_INPUT_DIM*indrow +
                  CONV2_INPUT_DIM*indcol + filcnl]; 
  return res;
}

float conv2_weight_ind(int filrow, int filcol, int filcnl, int fil,
                       OPENCL_FLOATP conv2_weights)
{
  /* conv2 weights is 5x5x32x64, row col cnl filter. */
  float res = 0.0;
  res = conv2_weights[CONV2_SIZE*CONV2_INPUT_DIM*CONV2_FILTER*filrow +
                      CONV2_INPUT_DIM*CONV2_FILTER*filcol +
                      CONV2_FILTER*filcnl + fil];
  return res;
}

float conv2_convolve(int row, int col, int fil, OPENCL_FLOATP conv2_W,
                     OPENCL_FLOATP conv2_b, float* pool1_out)
{
  float res = 0.0;
  /* CONV2 filter has 5x5x32 size, 64 filters*/
  #pragma unroll
  for (int filrow = 0; filrow < CONV2_SIZE; filrow++) {
    #pragma unroll
    for (int filcol = 0; filcol < CONV2_SIZE; filcol++) {
      #pragma unroll
      for (int filcnl = 0; filcnl < CONV2_INPUT_DIM; filcnl++) {
        res += conv2_input_ind(filrow, filcol, filcnl, row, col, pool1_out) *
               conv2_weight_ind(filrow, filcol, filcnl, fil, conv2_W);
      }
    }
  }
  res += conv2_b[fil];
  res = relu(res);
  return res;
}

/*--------------------MAIN KERNEL FUNCTION-------------------*/

// TODO: Build a CNN!
__attribute__((reqd_work_group_size(100,1,1))) // change this to change workgroup size
__kernel void linear_classifier(global const unsigned char * restrict images, 
								constant float * restrict conv1_weights,
								constant float * restrict conv1_bias,
								constant float * restrict conv2_weights,
								constant float * restrict conv2_bias,
								constant float * restrict dense1_weights,
								constant float * restrict dense1_bias,
								constant float * restrict dense2_weights,
								constant float * restrict dense2_bias,
								global unsigned char * restrict guesses)
{
  int img_arr_idx = get_global_id(0) * ARRAY_DIM;
  float conv1_out[28 * 28 * 32] = {0.0};
  float pool1_out[14 * 14 * 32] = {0.0};
  float conv2_out[14 * 14 * 64] = {0.0};
  float pool2_out[7 * 7 * 64] = {0.0};
  float dense1_out[DENSE1_SIZE] = {0.0};
  float dense2_out[DENSE2_SIZE] = {0.0};


	/* CONV LAYER 1 */
	#pragma unroll
  for (int row = 0; row < CONV1_OUT_SIDE; row++) {
    #pragma unroll
    for (int col = 0; col < CONV1_OUT_SIDE; col++) {
      #pragma unroll
      for (int fil = 0; fil < CONV1_FILTER; fil++) {
        conv1_out[CONV1_OUT_SIDE*CONV1_FILTER*row + CONV1_FILTER*col + fil] =
          conv1_convolve(row, col, fil, conv1_weights, conv1_bias, images, img_arr_idx);
      }
    }
  }


	/* MAXPOOL LAYER 1 */
  #pragma unroll
  for (int row = 0; row < POOL1_DIM; row++) {
    #pragma unroll
    for (int col = 0; col < POOL1_DIM; col++) {
      #pragma unroll
      for (int fil = 0; fil < CONV1_FILTER; fil++) {
        pool1_out[POOL1_DIM*CONV1_FILTER*row + CONV1_FILTER*col + fil] =
            largest_four(conv1_out[CONV1_OUT_SIDE*CONV1_FILTER*(2*row) + CONV1_FILTER*(2*col) + fil],
                         conv1_out[CONV1_OUT_SIDE*CONV1_FILTER*(2*row) + CONV1_FILTER*(2*col+1) + fil],
                         conv1_out[CONV1_OUT_SIDE*CONV1_FILTER*(2*row+1) + CONV1_FILTER*(2*col) + fil],
                         conv1_out[CONV1_OUT_SIDE*CONV1_FILTER*(2*row+1) + CONV1_FILTER*(2*col+1) + fil]);
      }
    }
  }

	/* CONV LAYER 2 */
  #pragma unroll
  for (int row = 0; row < CONV2_OUT_SIDE; row++) {
    #pragma unroll
    for (int col = 0; col < CONV2_OUT_SIDE; col++) {
      #pragma unroll
      for (int fil = 0; fil < CONV2_FILTER; fil++) {
        conv2_out[CONV2_OUT_SIDE*CONV2_FILTER*row + CONV2_FILTER*col + fil] =
          conv2_convolve(row, col, fil, conv2_weights, conv2_bias, pool1_out);
      }
    }
  }

	/* MAXPOOL LAYER 2 */
  #pragma unroll
  for (int row = 0; row < POOL2_DIM; row++) {
    #pragma unroll
    for (int col = 0; col < POOL2_DIM; col++) {
      #pragma unroll
      for (int fil = 0; fil < CONV2_FILTER; fil++) {
        pool2_out[POOL2_DIM*CONV2_FILTER*row + CONV2_FILTER*col + fil] =
            largest_four(conv2_out[CONV2_OUT_SIDE*CONV2_FILTER*(2*row) + CONV2_FILTER*(2*col) + fil],
                         conv2_out[CONV2_OUT_SIDE*CONV2_FILTER*(2*row) + CONV2_FILTER*(2*col+1) + fil],
                         conv2_out[CONV2_OUT_SIDE*CONV2_FILTER*(2*row+1) + CONV2_FILTER*(2*col) + fil],
                         conv2_out[CONV2_OUT_SIDE*CONV2_FILTER*(2*row+1) + CONV2_FILTER*(2*col+1) + fil]);
      }
    }
  }

	/* DENSE LAYER */
  #pragma unroll
  for (int r = 0; r < DENSE1_SIZE; r++) {
    #pragma unroll
    for (int l = 0; l < DENSE1_INPUT_DIM; l++) {
      dense1_out[r] += dense1_weights[l*DENSE1_SIZE+r] * pool2_out[l];
    }
    dense1_out[r] += dense1_bias[r];
    dense1_out[r] = relu(dense1_out[r]);
  }

	/* DENSE 2 */
  #pragma unroll
  for (int r = 0; r < DENSE2_SIZE; r++) {
    #pragma unroll
    for (int l = 0; l < DENSE1_SIZE; l++) {
      dense2_out[r] += dense2_weights[l*DENSE2_SIZE+r] * dense1_out[l];
    }
    dense2_out[r] += dense2_bias[r];
    dense2_out[r] = relu(dense2_out[r]);
  }
	
	/* FINAL GUESS */
  unsigned char guess = 0;
  for (unsigned char i = 1; i < 10; i++)
    if (dense2_out[i] > dense2_out[guess]) guess = i;
	guesses[get_global_id(0)] = guess;
}
