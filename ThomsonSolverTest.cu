
#include "ThomsonSolver.cuh"

#include <cuda_runtime.h>

template <typename T>
__device__ void calculate_tridiagonal_matrix(T const *f_dev, T const *d_dev, T *tsa_dev, T *tsb_dev, T *tsc_dev, T *tsd_dev, size_t size, T r) { // r = dt/(dx*dx)
	tsa_dev[0] = T(0); 
	tsb_dev[0] = d_dev[0] * r / 2 + T(1); 
	tsc_dev[0] = -d_dev[0] * r / 2;
	tsd_dev[0] = f_dev[0] + r / 2 * d_dev[0] * (f_dev[1] - f_dev[0]);

	for (size_t idx = 1; idx != size-2; ++idx) {
		tsa_dev[idx] = -r / 2 * d_dev[idx - 1];
		tsb_dev[idx] = r / 2 * (d_dev[idx - 1] + d_dev[idx]) + T(1);
		tsc_dev[idx] = -r / 2 * d_dev[idx];
		tsd_dev[idx] = f_dev[idx] + r / 2 * (f_dev[idx-1]*d_dev[idx-1] - f_dev[idx]*(d_dev[idx-1]+d_dev[idx]) + f_dev[idx+1]*d_dev[idx]);
	}

	tsa_dev[size - 2] = -r / 2 * d_dev[size - 3];
	tsb_dev[size - 2] = r / 2 * (d_dev[size - 3] + d_dev[size-2]) + T(1);
	tsc_dev[size - 2] = T(0);
	tsd_dev[size - 2] = f_dev[size - 2] + r / 2 * (f_dev[size - 3] * d_dev[size - 3] - f_dev[size - 2] * (d_dev[size - 3] + d_dev[size - 2]) + 2 * f_dev[size - 1] * d_dev[size - 2]);
}

template <typename T>
__device__ void diffusion_step(T *f, T *dfc, T *a, T *b, T *c, T *d, size_t size, T r) { // r = dt/(dx*dx)
	calculate_tridiagonal_matrix(f, dfc, a, b, c, d, size, r);
	iki::math::device::thomson_sweep(a, b, c, d, f, size - 1);
}

template <typename T>
__device__ void set_initial_state(T *f, T *d, size_t size) {
	T grad = T(1) / (size-1);
	f[0] = T(1); f[size - 1] = T(0);
	d[0] = d[size - 1] = T(1);
	for (size_t idx = 1; idx != size - 1; ++idx) {
		f[idx] = T(1) - grad * idx;
		d[idx] = T(1);
	}
}

template <typename T>
__global__ void thomson_sweep_test_kernell(T *mem, size_t size, size_t span, size_t loop_count) {
	size_t mem_shift = threadIdx.x * size * 6;
	T *f = mem + mem_shift;
	T *dfc = f + size, *a = f + 2 * size, *b = f + 3 * size, *c = f + 4 * size, *d = f + 5 * size;
	set_initial_state(f, dfc, size);
	for (; loop_count != 0; --loop_count) {
		diffusion_step(f,dfc,a,b,c,d,size,T(1.));
	}
}

#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

int main() {
	using namespace std;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "Error in starting cuda device" << endl;
		goto End;
	}

	{
		size_t size = 1024, span = 512;
		vector<float> f_next(size);

		//we need to allocate a number of grids size*span elements each
		//f_curr_dev, d_curr_dev
		//we also need to allocate 4 grids for the Thomson sweep method
		//a b c and d
		//6 grids in total
		float *mem_dev = NULL; //a pointer to the device global memory 
		if (cudaSuccess != cudaMalloc((void **)&mem_dev, 6 * size * span * sizeof(float))) {
			cout << "Can't allocate memory for function: " << 6 * size * span * sizeof(float) / 1024 << " Kb" << endl;
			goto Clear;
		}

		{
			auto begin = chrono::steady_clock::now(), end = begin;
			thomson_sweep_test_kernell <<<1,span>>> (mem_dev, size, span, 10u);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cout << "Kernell launch failed: " << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				goto Clear;
			}
			cudaDeviceSynchronize();
			end = chrono::steady_clock::now();
			cout << "Time consumed: "<< chrono::duration <double, milli>(end-begin).count() << " ms" << endl;
		}
		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Kernell execution failed: " << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		else {
			cout << "Calculation Success!" << endl;
			ofstream ascii_out("./data/f.txt"); ascii_out.precision(7); ascii_out.setf(std::ios::fixed, std::ios::floatfield);
			for (size_t row = 0; row != span; ++row) {
				if (cudaSuccess != cudaMemcpy(f_next.data(), mem_dev + 6 * size * row, size * sizeof(float), cudaMemcpyDeviceToHost)) {
					cout << "Memory copy device->host failed!" << endl;
				}
				else {
					for (auto f : f_next) {
						ascii_out << f << ' ';
					}
					ascii_out << endl;
				}
			}
		}


	Clear:;
		if (mem_dev != NULL) cudaFree(mem_dev);
	}

	if (cudaSuccess != cudaDeviceReset()) {
		cout << "Error in device process termination!" << endl;
	}
End:;
	return 0;
}