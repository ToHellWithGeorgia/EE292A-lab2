// TODO: Define any constants you'll need
#define ARRAY_DIM 784
#define CONV1_SIZE 5
#define CONV1_INPUT_DIM 1
#define CONV1_FILTER 32
#define CONV1_OUT_SIDE 28
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
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

// TODO: If you decide you'd like to write helper functions, you can define them here
float largest_four(float a, float b, float c, float d)
{
  float ret = a;
  if (b > ret) ret = b;
  if (c > ret) ret = c;
  if (d > ret) ret = d;
  return ret;
}

float relu(float in)
{
  return in > 0 ? in : 0;
}



// TODO: Build a CNN!
__attribute__((reqd_work_group_size(10000,1,1))) // change this to change workgroup size
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

	/* MAXPOOL LAYER 1 */
  for (int row = 0; row < POOL1_DIM; row++) {
    for (int col = 0; col < POOL1_DIM; col++) {
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

	/* MAXPOOL LAYER 2 */
  for (int row = 0; row < POOL2_DIM; row++) {
    for (int col = 0; col < POOL2_DIM; col++) {
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
  for (int r = 0; r < DENSE1_SIZE; r++) {
    for (int l = 0; l < DENSE1_INPUT_DIM; l++) {
      dense1_out[r] += dense1_weights[l*DENSE1_SIZE+r] * pool2_out[l];
    }
    dense1_out[r] += dense1_bias[r];
    dense1_out[r] = relu(dense1_out[r]);
  }

	/* DENSE 2 */
  for (int r = 0; r < DENSE2_SIZE; r++) {
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
	guesses[get_global_id(0)] = 0;
}
