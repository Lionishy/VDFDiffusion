#include "ForwardStepKernel.cuh"
#include "ThomsonSolverKernel.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

template <typename T>
void initial_value(std::vector<T> &f, size_t x_size, size_t y_size) {
	T grad = T(1) / (x_size-1);
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx) {
			f[x_idx*y_size + y_idx] = T(1) - grad * x_idx;
		}
}

template <typename T>
void x_initial_diffusion(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		dfc[y_idx] = dfc[(x_size-1)*y_size + y_idx] = T(0);
	
	for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
		dfc[x_idx * y_size] = dfc[y_size-1 + x_idx*y_size] = T(0);

	for (size_t x_idx = 1; x_idx != x_size - 1; ++x_idx)
		for (size_t y_idx = 1; y_idx != y_size - 1; ++y_idx)
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
	vector<float> f(x_size * y_size), x_diffusion(x_size * y_size), y_diffusion(x_size * y_size);
	initial_value(f, x_size, y_size); x_initial_diffusion(x_diffusion, x_size, y_size); y_initial_diffusion(y_diffusion, x_size, y_size);
	vector<float> host_a((x_size - 2) * (y_size - 2));
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

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + x_size*y_size, f.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
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

		int blockDim = 1, threads = 1022;
		auto begin = chrono::steady_clock::now(), end = begin;
		for (int count = 0; count != 10000; ++count) {
			diffusion::device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, a, b, c, d, rx, ry, x_size - 2, y_size /*x_sitride*/, x_size /*y_stride*/);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cout << "Kernel launch failed!" << endl;
				cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_prev, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cout << "Kernel launch failed!" << endl;
				cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}
		}

		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;

		if (cudaSuccess != (cudaStatus = cudaMemcpy(f.data(), gm_dev, x_size*y_size * sizeof(float), cudaMemcpyDeviceToHost))) {
			cout << "Can't copy data from f_prev to host:" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		else {
			ofstream ascii_out("./data/matrix.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					ascii_out << x_idx << " " << y_idx << " " << f[x_idx * y_size + y_idx] << endl;
		}
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