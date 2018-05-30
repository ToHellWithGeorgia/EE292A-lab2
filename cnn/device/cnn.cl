// TODO: Define any constants you'll need
#define ARRAY_DIM 784
#define CONV1_SIZE 5
#define CONV1_INPUT_DIM 1
#define CONV1_FILTER 32
#define CONV2_SIZE 5
#define CONV2_INPUT_DIM 32
#define CONV2_FILTER 64
#define DENSE1_SIZE 256
#define DENSE1_INPUT_DIM (7*7*64)
#define DENSE2_SIZE 10
#define DENSE2_INPUT_DIM (256*1*1)
// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)

// TODO: If you decide you'd like to write helper functions, you can define them here


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
  float conv1_out[28 * 28 *32];
  float pool1_out[14 * 14 * 32];
  float conv2_out[14 * 14 * 64];
  float pool2_out[7 * 7 * 64];
  float dense1_out[DENSE1_SIZE];
  float dense2_out[DENSE2_SIZE];

	/* CONV LAYER 1 */

	/* MAXPOOL LAYER 1 */

	/* CONV LAYER 2 */

	/* MAXPOOL LAYER 2 */

	/* DENSE LAYER */

	/* DENSE 2 */
	
	/* FINAL GUESS */
	guesses[get_global_id(0)] = 0;
}
