#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
#include <fstream>
#include "assert.h"
// #include <time.h>
// #include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../../shared/defines.h"
#include "../../shared/utils.h"

// TODO: If you want to define constants, you can do it here
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

using namespace aocl_utils;

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;

cl_uchar *input_images = NULL, *output_guesses = NULL, *reference_guesses = NULL;
cl_float *input_weights = NULL;
cl_mem input_images_buffer, output_guesses_buffer;
// TODO: add buffers for your weights
cl_float *conv1_W = NULL, *conv1_b = NULL;
cl_float *conv2_W = NULL, *conv2_b = NULL;
cl_float *dense1_W = NULL, *dense1_b = NULL;
cl_float *dense2_W = NULL, *dense2_b = NULL;
cl_mem conv1_W_buffer, conv1_b_buffer, conv2_W_buffer, conv2_b_buffer;
cl_mem dense1_W_buffer, dense1_b_buffer, dense2_W_buffer, dense2_b_buffer;

// Global variables.
std::string imagesFilename;
std::string labelsFilename;
std::string aocxFilename;
std::string deviceInfo;
unsigned int n_items;
bool use_fixed_point;
bool use_single_workitem;

// Function prototypes.
void classify();
void initCL();
void cleanup();
void teardown(int exit_status = 1);



int main(int argc, char **argv) {
	// Parsing command line arguments.
	Options options(argc, argv);

	if(options.has("images")) {
		imagesFilename = options.get<std::string>("images");
	} else {
		imagesFilename = "t10k-images.idx3-ubyte";
		printf("Defaulting to images file \"%s\"\n", imagesFilename.c_str());
	}
	
	if(options.has("labels")) {
		labelsFilename = options.get<std::string>("labels");  
	} else {
		labelsFilename = "t10k-labels.idx1-ubyte";
		printf("Defaulting to labels file \"%s\"\n", labelsFilename.c_str());
	}

	// Relative path to aocx filename option.
	if(options.has("aocx")) {
		aocxFilename = options.get<std::string>("aocx");  
	} else {
		aocxFilename = "linear_classifier_fp";
		printf("Defaulting to aocx file \"%s\"\n", aocxFilename.c_str());
	}
	
	// Read in the images and labels
	n_items = parse_MNIST_images(imagesFilename.c_str(), &input_images);
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		return -1;
	}
	if (n_items != parse_MNIST_labels(labelsFilename.c_str(), &reference_guesses)){
		printf("ERROR: Number of labels does not match number of images\n");
		return -1;
	}

	// TODO: Uncomment this to verify on a smaller set of examples
	n_items = 100;
	
	// Initializing OpenCL and the kernels.
	output_guesses = (cl_uchar*)alignedMalloc(sizeof(cl_uchar) * n_items);
	// TODO: Allocate space for weights if you so desire. To help you out, here's the declaration from last time:	
	// input_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * FEATURE_COUNT * NUM_DIGITS);
  int conv1_W_num = CONV1_SIZE * CONV1_SIZE * CONV1_INPUT_DIM * CONV1_FILTER;
  int conv1_b_num = CONV1_FILTER;
  int conv2_W_num = CONV2_SIZE * CONV2_SIZE * CONV2_INPUT_DIM * CONV2_FILTER;
  int conv2_b_num = CONV2_FILTER;
  int dense1_W_num = DENSE1_SIZE * DENSE1_INPUT_DIM;
  int dense1_b_num = DENSE1_SIZE;
  int dense2_W_num = DENSE2_SIZE * DENSE2_INPUT_DIM;
  int dense2_b_num = DENSE2_SIZE;
  
  conv1_W = (cl_float*)alignedMalloc(sizeof(cl_float) * conv1_W_num);
  conv1_b = (cl_float*)alignedMalloc(sizeof(cl_float) * conv1_b_num);
  conv2_W = (cl_float*)alignedMalloc(sizeof(cl_float) * conv2_W_num);
  conv2_b = (cl_float*)alignedMalloc(sizeof(cl_float) * conv2_b_num);
  dense1_W = (cl_float*)alignedMalloc(sizeof(cl_float) * dense1_W_num);
  dense1_b = (cl_float*)alignedMalloc(sizeof(cl_float) * dense1_b_num);
  dense2_W = (cl_float*)alignedMalloc(sizeof(cl_float) * dense2_W_num);
  dense2_b = (cl_float*)alignedMalloc(sizeof(cl_float) * dense2_b_num);

	
	// TODO: Read in weights from weights files
  char conv1_W_name[256] = "weights/conv1_weights";
  char conv1_b_name[256] = "weights/conv1_bias";
  char conv2_W_name[256] = "weights/conv2_weights";
  char conv2_b_name[256] = "weights/conv2_bias";
  char dense1_W_name[256] = "weights/dense1_weights";
  char dense1_b_name[256] = "weights/dense1_bias";
  char dense2_W_name[256] = "weights/dense2_weights";
  char dense2_b_name[256] = "weights/dense2_bias";
  bool status = false;
  
  status = read_weights_file(conv1_W_name, conv1_W, conv1_W_num);
  assert(status);
  status = read_weights_file(conv1_b_name, conv1_b, conv1_b_num);
  assert(status);
  status = read_weights_file(conv2_W_name, conv2_W, conv2_W_num);
  assert(status);
  status = read_weights_file(conv2_b_name, conv2_b, conv2_b_num);
  assert(status);
	
  status = read_weights_file(dense1_W_name, dense1_W, dense1_W_num);
  // printf("Sanity: dense1_w_0: %f\n", dense1_W[0]);
  assert(status);
  status = read_weights_file(dense1_b_name, dense1_b, dense1_b_num);
  assert(status);
  status = read_weights_file(dense2_W_name, dense2_W, dense2_W_num);
  assert(status);
  status = read_weights_file(dense2_b_name, dense2_b, dense2_b_num);
  assert(status);
	
	initCL();

	// Start measuring time
	double start = get_wall_time();
	
	// Call the classifier.
	classify();
	
	// Stop measuring time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
   
	int correct = 0;
	for (unsigned i = 0; i < n_items; i++){
		if (output_guesses[i] == reference_guesses[i]) correct++;
	}
	printf("Classifier accuracy: %.2f%%\n", (float)correct*100/n_items);
	
	// Teardown OpenCL.
	teardown(0);
}

void classify() {
  int conv1_W_num = CONV1_SIZE * CONV1_SIZE * CONV1_INPUT_DIM * CONV1_FILTER;
  int conv1_b_num = CONV1_FILTER;
  int conv2_W_num = CONV2_SIZE * CONV2_SIZE * CONV2_INPUT_DIM * CONV2_FILTER;
  int conv2_b_num = CONV2_FILTER;
  int dense1_W_num = DENSE1_SIZE * DENSE1_INPUT_DIM;
  int dense1_b_num = DENSE1_SIZE;
  int dense2_W_num = DENSE2_SIZE * DENSE2_INPUT_DIM;
  int dense2_b_num = DENSE2_SIZE;

	size_t size = 1;
	cl_int status;
	cl_event event;
	const size_t global_work_size = n_items;
	
	// Create kernel input and output buffers.
	// TODO: Add buffers for layer weights
	input_images_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * FEATURE_COUNT * n_items, NULL, &status);
	checkError(status, "Error: could not create input image buffer");
	output_guesses_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * n_items, NULL, &status);
	checkError(status, "Error: could not create output guesses buffer");

  conv1_W_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * conv1_W_num, NULL, &status);
	checkError(status, "Error: could not create conv1_W buffer");
  conv1_b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * conv1_b_num, NULL, &status);
	checkError(status, "Error: could not create conv1_b buffer");
  conv2_W_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * conv2_W_num, NULL, &status);
	checkError(status, "Error: could not create conv2_W buffer");
  conv2_b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * conv2_b_num, NULL, &status);
	checkError(status, "Error: could not create conv2_b buffer");
  dense1_W_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * dense1_W_num, NULL, &status);
	checkError(status, "Error: could not create dense1_W buffer");
  dense1_b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * dense1_b_num, NULL, &status);
	checkError(status, "Error: could not create dense1_b buffer");
  dense2_W_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * dense2_W_num, NULL, &status);
	checkError(status, "Error: could not create dense2_W buffer");
  dense2_b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * dense2_b_num, NULL, &status);
	checkError(status, "Error: could not create dense2_b buffer");

	
	// Copy data to kernel input buffer.
	// TODO: Copy weights for your layers as well
	status = clEnqueueWriteBuffer(queue, input_images_buffer, CL_TRUE, 0, sizeof(unsigned char) * FEATURE_COUNT * n_items, input_images, 0, NULL, NULL);
	checkError(status, "Error: could not copy images into device");

  status = clEnqueueWriteBuffer(queue, conv1_W_buffer, CL_TRUE, 0, sizeof(cl_float) * conv1_W_num, conv1_W, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv1_W into device");
  status = clEnqueueWriteBuffer(queue, conv1_b_buffer, CL_TRUE, 0, sizeof(cl_float) * conv1_b_num, conv1_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv1_b into device");
  status = clEnqueueWriteBuffer(queue, conv2_W_buffer, CL_TRUE, 0, sizeof(cl_float) * conv2_W_num, conv2_W, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv2_W into device");
  status = clEnqueueWriteBuffer(queue, conv2_b_buffer, CL_TRUE, 0, sizeof(cl_float) * conv2_b_num, conv2_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy conv2_b into device");

  status = clEnqueueWriteBuffer(queue, dense1_W_buffer, CL_TRUE, 0, sizeof(cl_float) * dense1_W_num, dense1_W, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense1_W into device");
  status = clEnqueueWriteBuffer(queue, dense1_b_buffer, CL_TRUE, 0, sizeof(cl_float) * dense1_b_num, dense1_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense1_b into device");
  status = clEnqueueWriteBuffer(queue, dense2_W_buffer, CL_TRUE, 0, sizeof(cl_float) * dense2_W_num, dense2_W, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense2_W into device");
  status = clEnqueueWriteBuffer(queue, dense2_b_buffer, CL_TRUE, 0, sizeof(cl_float) * dense2_b_num, dense2_b, 0, NULL, NULL);
	checkError(status, "Error: could not copy dense2_b into device");
	
	// Set the arguments for data_in, data_out and sobel kernels.
	// TODO: Set arguments for your weights
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_images_buffer);
	checkError(status, "Error: could not set argument 0");

  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&conv1_W_buffer);
	checkError(status, "Error: could not set argument 1");
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&conv1_b_buffer);
	checkError(status, "Error: could not set argument 2");
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&conv2_W_buffer);
	checkError(status, "Error: could not set argument 3");
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&conv2_b_buffer);
	checkError(status, "Error: could not set argument 4");
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&dense1_W_buffer);
	checkError(status, "Error: could not set argument 5");
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&dense1_b_buffer);
	checkError(status, "Error: could not set argument 6");
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&dense2_W_buffer);
	checkError(status, "Error: could not set argument 7");
  status = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&dense2_b_buffer);
	checkError(status, "Error: could not set argument 8");

	status = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&output_guesses_buffer);
	checkError(status, "Error: could not set argument 9");

	
	// Enqueue the kernel. //
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event);
	checkError(status, "Error: failed to launch data_in");
	
	// Wait for command queue to complete pending events.
	printf("Waiting for kernel to finish..?\n");
	status = clFinish(queue);
	printf("Kernel has finished\n");
	checkError(status, "Kernel failed to finish");

	clReleaseEvent(event);
	
	// Read output buffer from kernel.
	status = clEnqueueReadBuffer(queue, output_guesses_buffer, CL_TRUE, 0, sizeof(unsigned char) * n_items, output_guesses, 0, NULL, NULL);
	checkError(status, "Error: could not copy data from device");
}

void initCL() {
	cl_int status;

	// Start everything at NULL to help identify errors.
	kernel = NULL;
	queue = NULL;
	
	// Locate files via. relative paths.
	if(!setCwdToExeDir()) {
		teardown();
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA");
	if (platform == NULL) {
		teardown();
	}

	// Get the first device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError (status, "Error: could not query devices");

	char info[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
	deviceInfo = info;

	// Create the context.
	context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Error: could not create OpenCL context");

	// Create the command queues for the kernels.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
	std::cout << "Using AOCX: " << binary_file << "\n";
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 1, &device, "", NULL, NULL);
	checkError(status, "Error: could not build program");
	
	// Create the kernel - name passed in here must match kernel name in the original CL file.
	kernel = clCreateKernel(program, "linear_classifier", &status);
	checkError(status, "Failed to create kernel");
}

void cleanup() {
	// Called from aocl_utils::check_error, so there's an error.
	teardown(-1);
}

void teardown(int exit_status) {
	if(kernel) clReleaseKernel(kernel);
	if(queue) clReleaseCommandQueue(queue);
	if(input_images) alignedFree(input_images);
	if(input_weights) alignedFree(input_weights);
  if(conv1_W) alignedFree(conv1_W);
  if(conv1_b) alignedFree(conv1_b);
  if(conv2_W) alignedFree(conv2_W);
  if(conv2_b) alignedFree(conv2_b);
  if(dense1_W) alignedFree(dense1_W);
  if(dense1_b) alignedFree(dense1_b);
  if(dense2_W) alignedFree(dense2_W);
  if(dense2_b) alignedFree(dense2_b);
	if(reference_guesses) alignedFree(reference_guesses);
	if(output_guesses) alignedFree(output_guesses);
	if(input_images_buffer) clReleaseMemObject(input_images_buffer);
	if(output_guesses_buffer) clReleaseMemObject(output_guesses_buffer);
	if(program) clReleaseProgram(program);
	if(context) clReleaseContext(context);
	
	exit(exit_status);
}
