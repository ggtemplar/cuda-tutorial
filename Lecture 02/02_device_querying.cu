/* Querying the GPU for various details like memory, compute capability, etc. */

/* Header files. */
#include <stdio.h>

/* main */
int main (void)
{

	/* Discussion of individual variables might be covered in advanced tutorial.
	 * +--------------------------------------------------------------------+
	 * | struct cudaDeviceProp                                              |
	 * +--------------------------------------------------------------------+
	 * | Will hold the device properties.                                   |
	 * +--------------------------+-----------------------------------------+
	 * | Variable                 | Description                             |
	 * +--------------------------+-----------------------------------------+
	 * | char name[265]           | Name of the GPU.                        |
	 * | size_t totalGlobalMem    | Total amount of global memory available |
	 * |                          | on device in bytes.                     |
	 * | size_t sharedMemPerBlock | Maximum amount of shared memory         |
	 * |                          | available to a thread block in bytes;   |
	 * |                          | this amount is shared by all thread     |
	 * |                          | blocks simultaneously resident on a     |
	 * |                          | multiprocessor.                         |
	 * | int regsPerBlock         | Maximum number of 32-bit registers      |
	 * |                          | available to a thread block. this number|
	 * |                          | is shared by all threads simultaneously |
	 * |                          | resident on a multiprocessor.           |
	 * | int warpSize             | The warp size is threads or the number  |
	 * |                          | of threads that a processor can execute |
	 * |                          | concurrently.                           |
	 * | size_t memPitch          | Maximum pitch size in bytes.            |
	 * | int maxThreadsPerBlock   | Maximum threads per block.              |
	 * | int maxThreadsDim[3]     | Maximum size of each dimension of a     |
	 * |                          | block.                                  |
	 * | int maxGridSize[3]       | Contains maximum size of each dimension |
	 * |                          | of a grid.                              |
	 * | int major, minor         | Device's compute capability.            |
	 * | int clockRate            | Clock frequency in KHz.                 |
	 * | size_t textureAlignment  | The aligned texture base addresses do   |
	 * |                          | not need an offset for texture fetches. |
	 * | size_t totalConstMem     | Total amount of constant memory on GPU. |
	 * | int deviceOverlap        | Set to 1 when device can concurrently   |
	 * |                          | copy memory between host and device     |
	 * |                          | while executing a kernel.               |
	 * | int muliProcessorCount   | Number of multiprocessors on the device |
	 * | int kernelExecuteTimeoutEnabled | Set to 1 if there is a runtime   |
	 * |                          | limit for kernels executed on device.   |
	 * | int integrated           | Set to 1 if the GPU is integrated with  |
	 * |                          | the motherboard.                        |
	 * | int canMapHostMemory     | Set to 1 if device can map host memory  |
	 * |                          | into CUDA address space.                |
	 * | int computeMode          | -> `default`: Device is not restricted  |
	 * | cudaComputeModeDefault   | and multiple threads can use            |
	 * |                          | `cudaSetDevice()` for this device.      |
	 * | cudaComputeModeExclusive | -> `Compute-exclusive mode`: Only one   |
	 * |                          | thread will be ale to use               |
	 * |                          | `cudaSetDevice()` with this device.     |
	 * | cudaComputeModeProhibited| -> `Compute-prohibited mode`: No threads|
	 * |                          | can use `cudaSetDevice()` with this     |
	 * |                          | device.                                 |
	 * | int concurrentKernels    | Set to 1 if the device supports multiple|
	 * |                          | kernels within the same context         |
	 * |                          | simultaneously.                         |
	 * | int ECCEnabled           | Set to 1 if device has ECC support      |
	 * |                          | turned on.                              |
	 * | int pciBusID             | PCI Bus identifier for the device.      |
	 * | int pciDeviceID          | Slot identifier of the device.          |
	 * | int tccDriver            | Set to 1 if the device is using TCC     |
	 * |                          | driver.                                 |
	 * +--------------------------+-----------------------------------------+ */
	cudaDeviceProp prop;

	/* Maintain the count of devices. */
	int count;


	/* +-----------------------------------------------------------------------+
	 * | cudaError_t cudaGetDeviceCount ( int* count )                         |
	 * +-----------------------------------------------------------------------+
	 * | Returns in `*count`, number of devices with compute capability        |
	 * | greater than or equal to 1.0 In case the device supports emulation    |
	 * | mode, count becomes 1.                                                |
	 * +----------+------------------------------------------------------------+
	 * |Arguments | Address of count, since count holds the number of devices. |
	 * +----------+------------------------------------------------------------+ */
	cudaGetDeviceCount (&count);

	/* Iterate over all the devices, retrieving their properties one by one. */
	for ( int i = 0; i < count; i++ )
	{
		/* +--------------------------------------------------------------------+
		 * | cudaError_t cudaGetDeviceProperties ( struct cudaDeviceProp *prop, |
		 * |                                       int device )                 |
		 * +--------------------------------------------------------------------+
		 * | Returns in `*prop` the device properties of `dev`.                 |
		 * +--------------------------------------------------------------------+ */
		cudaGetDeviceProperties (&prop, i);

		/* Displaying some of these properties. */
		printf ("\n--------------------DEVICE-PROPERTIES--------------------");
		printf ("\nName: NVIDIA %s", prop.name);
		printf ("\nTotal Global Memory: %.1f MB",
				1.0 * prop.totalGlobalMem / (1024*1024));
		printf ("\nShared Memory per Block: %.1f KB",
				1.0 * prop.sharedMemPerBlock / 1024);
		printf ("\nCompute Capability: %d.%d", prop.major, prop.minor);
		printf ("\nClock Frequency: %.2f GHz", 1.0 * prop.clockRate / 1000000);
		printf ("\nConcurrent Kernels: %s",
				prop.concurrentKernels ? "Enabled" : "Disabled" );
		printf ("\nMultiprocessors: %d", prop.multiProcessorCount);

		return 0;
	}
}
