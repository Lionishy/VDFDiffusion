#include "ForwardStepKernel.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

template <typename T>
void initial_value(std::vector<T> &matrix, size_t x_size, size_t y_size) {
	T grad = T(1) / (x_size-1);
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx) {
			matrix[x_idx*y_size + y_idx] = T(1) - grad * x_idx;
		}
}

template <typename T>
void x_initial_diffusion(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		dfc[y_idx] = dfc[(x_size-1)*y_size + y_idx] = T(0);
	
	for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
		dfc[x_idx * y_size] = dfc[y_size-1 + x_idx*y_size] = T(0);

	for (size_t x_idx = 1; x_idx != x_size - 2; ++x_idx)
		for (size_t y_idx = 1; y_idx != y_size - 2; ++y_idx)
			dfc[x_idx * y_size + y_idx] = T(1);
}

template <typename T>
void y_initial_diffusion(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
		for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
			dfc[x_idx * y_size + y_idx] = T(0);
}

int main() {
	using namespace std;
	using namespace iki;
	
	/**
	 * Loop over...
	 * ForwardCalculationKernell -> ThomsonKernell -> TransposeKernell -> CorrectionCalculationKernell -> ThomsonKernell -> TransposeKernell
	 */

	cudaError_t cudaStatus;
	float *gm_dev = NULL;
	size_t x_size = 1024, y_size = 1024;
	vector<float> f(x_size * y_size), x_diffusion(x_size*y_size), y_diffusion(x_size*y_size);
	initial_value(f, x_size, y_size); x_initial_diffusion(x_diffusion, x_size, y_size); y_initial_diffusion(y_diffusion, x_size, y_size);
	float rx = 1.f, ry = 1.f;

	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}

	if (cudaSuccess != (cudaStatus = cudaMalloc((void **)&gm_dev, 8 * x_size * y_size * sizeof(float)))) {
		cerr << "Can't allocate global device memory of " << (8*x_size*y_size*sizeof(float)/1024) << " Kb: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}
	else {
		cerr << (8 * x_size * y_size * sizeof(float) / 1024) << " Kb: " << " successfully allocated!" << endl;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev, f.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cout << "Can't copy data from f to device:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + 2*x_size*y_size, x_diffusion.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cout << "Can't copy data from x_dfc to device:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + 3 * x_size * y_size, y_diffusion.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cout << "Can't copy data from y_dfc to device:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	{
		size_t matrix_size = x_size * y_size;
		float *f_prev = gm_dev + y_size + 1, *f_curr = f_prev + matrix_size, *x_dfc = f_curr + matrix_size, *y_dfc = x_dfc + matrix_size, *a = y_dfc + matrix_size, *b = a + matrix_size, *c = b + matrix_size, *d = c + matrix_size;

		int blockDim = 1, threads = 1024;
		diffusion::device::forward_step_multisolver_kernel<<<blockDim,threads>>>(f_prev, x_dfc, y_dfc, a, b, c, d, rx, ry, x_size - 2, y_size /*x_sitride*/, x_size /*y_stride*/);
	}

Clear:
	if (NULL != gm_dev) cudaFree(gm_dev);
	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cerr << "Error in device process termination: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}

End:
	return 0;
}