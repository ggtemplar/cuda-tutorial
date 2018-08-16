/* Addition of two numbers using a kernel method.
 * Note: Documentation will explain each thing only once. */

/* Header files */
#include <stdio.h>


/* This a kernel function, it has the __global__ qualifier in the definition.
 * addition: Perform the addition of two numbers and return their sum.
 * +------------+-----------------------------------+
 * | Parameters | Description                       |
 * +------------+-----------------------------------+
 * | int  a     | Takes an integer passed by value. |
 * | int  b     | Takes an integer passed by value. |
 * | int *c     | An integer pointer that refers to |
 * |            | the GPU memory where we store the |
 * |            | result of the addition.           |
 * +------------+-----------------------------------+ */
__global__ void addition ( int a, int b, int *c )
{
	*c = a + b;
}


/* main method runs on the host. */
int main ( void )
{

	/* +----------------+------------------------------------+
	 * | Local Variable | Description                        |
	 * +----------------+------------------------------------+
	 * | int  c         | To store the result on the host.   |
	 * | int *c_dev     | A pointer of the host containing   |
	 * |                | the address to the pointer on the  |
	 * |                | GPU that will point to some memory |
	 * |                | which has our result.              |
	 * +----------------+------------------------------------+ */
	int c;
	int *c_dev;

	/* +----------------------------------------------------------+
	 * | cudaError_t cudaMalloc ( void** devPtr, size_t size)     |
	 * +----------------+-----------------------------------------+
	 * | Parameters     | Description                             |
	 * +----------------+-----------------------------------------+
	 * | void** devPtr | The address of the pointer to the       |
	 * |                | allocated memory on the GPU is returned |
	 * |                | to this pointer.                        |
	 * | size_t size    | The number of bytes to be allocated on  |
	 * |                | the GPU memory.                         |
	 * +----------------+-----------------------------------------+
	 * | Return Type    | typedef enum cudaError_t;               |
	 * +----------------+-----------------------------------------+
	 * | Can return `cudaSuccess` on a successful allocation and  |
	 * | `cudaErrorMemoryAllocation` on an unsuccessful one.      |
	 * +----------------+-----------------------------------------+
	 * | Arguments      | Description                             |
	 * +----------------+-----------------------------------------+
	 * | (void**)&c_dev | The address to the pointer c_dev,       |
	 * |                | casted to `(void **)`. This is a        |
	 * |                | pointer being passed by reference.      |
	 * | sizeof(int)    | The size of integer on the machine. We  |
	 * |                | require that much memory to store the   |
	 * |                | result of addition.                     |
	 * +----------------+-----------------------------------------+ */
	cudaMalloc ( (void **) &c_dev, sizeof (int) );

	/* Call to the kernel function. */
	addition <<< 1, 1 >>> ( 2, 8, c_dev);

	/* +----------------------------------------------------------+
	 * | cudaError_t cudaMemcpy ( void* dst, const void* src,     |
	 * |                          size_t count,                   |
	 * |                          enum cudaMemcpyKind kind )      |
	 * +-----------------+----------------------------------------+
	 * | Parameters      | Description                            |
	 * +-----------------+----------------------------------------+
	 * | void* dst       | Destination memory address.            |
	 * | const void* src | Source memory address.                 |
	 * | size_t count    | Size in bytes to copy.                 |
	 * | ... kind        | Type / direction of transfer. This can |
	 * |                 | be done in three ways:                 |
	 * |                 | -> `cudaMemcpyDeviceToHost`            |
	 * |                 | -> `cudaMemcpyDeviceToDevice`          |
	 * |                 | -> `cudaMemcpyHostToDevice`            |
	 * |                 | Getting the pointers or the direction  |
	 * |                 | wrong results in undefined behaviour.  |
	 * +-----------------+----------------------------------------+
	 * | Returns         | `cudaSuccess`, `cudaErrorInvalidValue` |
	 * |                 | `cudaErrorInvalidDevicePointer`,       |
	 * |                 | `cudaErrorInvalidMemcpyDirection`      |
	 * +-----------------+----------------------------------------+ */
	cudaMemcpy ( &c, c_dev, sizeof(int), cudaMemcpyDeviceToHost );

	/* Print our result. */
	printf("2 + 8 = %d", c);

	/* +----------------------------------------------------------+
	 * | cudaError_t cudaFree ( void** devPtr)                    |
	 * +----------------------------------------------------------+
	 * | Frees the memory pointed to by the device pointer.       |
	 * | Gives an error if try to deallocate freed space.         |
	 * +----------------------------------------------------------+ */
	cudaFree(c_dev);

	/* Normal exit. */
	return 0;
}
