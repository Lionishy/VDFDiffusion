
#include "ThomsonSolver.cuh"

#include <cuda_runtime.h>


template <typename T>
__global__ void set_test_matrix(T *mem_dev, size_t size) {
	T *a = mem_dev, *b = mem_dev + size, *c = mem_dev + 2 * size, *d = mem_dev + 3 * size, *x = mem_dev + 4*size;
	a[0] = 0.f; b[0] = 3.f; c[0] = 1.f; d[0] = 4.f;
	for (size_t idx = 1; idx != size-1; ++idx) {
		a[idx] = 1.f; b[idx] = 2.f; c[idx] = 1.f; d[idx] = 4.f;
	}
	a[size - 1] = 1.f; b[size - 1] = 3.f; c[size - 1] = 0.f; d[size - 1] = 4.f;

	for (size_t idx = 0; idx != size; ++idx)
		x[idx] = -1.f;
}

template <typename T>
__global__ void thomson_sweep_test_kernell(T *mem_dev, size_t size) {
	iki::math::device::thomson_sweep(mem_dev, mem_dev + size, mem_dev + 2 * size, mem_dev + 3 * size, mem_dev + 4 * size, size);
}

#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	using namespace std;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "Error in starting cuda device" << endl;
		goto End;
	}

	//cuda matrix data
	size_t size = 1000;
	double *mem_dev = NULL;// 5 * size =>  *a_dev, *b_dev, *c_dev, *d_dev, *x_dev;
	if (cudaSuccess != cudaMalloc((void **)&mem_dev, 5 * size * sizeof(double))) {
		cout << "Can't allocate enought device memory!" << endl;
		goto Clear;
	}

	set_test_matrix<<<1,1>>>(mem_dev, size);
	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaStatus) {
		cout << "Kernell launch failed: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	thomson_sweep_test_kernell<<<1,1>>>(mem_dev, size);
	cudaDeviceSynchronize();
	{
		vector<double> result(size);
		if (cudaSuccess != cudaMemcpy(result.data(), mem_dev + 4 * size, size * sizeof(double), cudaMemcpyDeviceToHost)) {
			cout << "Error while memory copy from device to host!" << endl;
			goto Clear;
		}

		for_each(begin(result), end(result), [] (auto x) { std::cout << x << '\n'; });
	}

Clear:;
	if (NULL != mem_dev) cudaFree(mem_dev);
	if (cudaSuccess != cudaDeviceReset()) {
		cout << "Error in device process termination!" << endl;
	}


End:;
	return 0;
}