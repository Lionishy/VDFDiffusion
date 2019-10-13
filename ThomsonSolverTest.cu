
#include "ThomsonSolver.cuh"

#include <cuda_runtime.h>

template <typename T>
__device__ void calculate_tridiagonal_matrix(T const *f_dev, T const *d_dev, T *tsa_dev, T *tsb_dev, T *tsc_dev, T *tsd_dev, size_t size, T r) {
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
__device__ void set_initial_state(T *f_dev, T *d_dev, size_t size) {
	T grad = T(1) / (size-1);
	f_dev[0] = T(1); f_dev[size - 1] = T(0);
	d_dev[0] = d_dev[size - 1] = T(1);
	for (size_t idx = 1; idx != size - 1; ++idx) {
		f_dev[idx] = T(1) - grad * idx;
		d_dev[idx] = T(1);
	}
}

template <typename T>
__global__ void thomson_sweep_test_kernell(T *f_dev, T *d_dev, T *tsa_dev, T *tsb_dev, T *tsc_dev, T *tsd_dev, T *tsx_dev, size_t size, size_t loop_count) {
	set_initial_state(f_dev, d_dev, size);
	for (; loop_count != 0; --loop_count) {
		calculate_tridiagonal_matrix(f_dev, d_dev, tsa_dev, tsb_dev, tsc_dev, tsd_dev, size, T(1.));
		iki::math::device::thomson_sweep(tsa_dev, tsb_dev, tsc_dev, tsd_dev, tsx_dev, size-1);
		for (size_t idx = 0; idx != size - 1; ++idx)
			f_dev[idx] = tsx_dev[idx];
	}
}

#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <fstream>

int main() {
	using namespace std;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "Error in starting cuda device" << endl;
		goto End;
	}

	{
		size_t size = 1024;
		vector<float> f_next(size);

		//cuda function data
		float *f_dev = NULL, *d_dev = NULL;
		//cuda thomson sweep method data
		float *tsa_dev = NULL, *tsb_dev = NULL, *tsc_dev = NULL, *tsd_dev = NULL, *tsx_dev = NULL;

		if (cudaSuccess != cudaMalloc((void **)&f_dev, size * sizeof(float))) {
			cout << "Can't allocate memory for function: " << size * sizeof(float) / 1024 << " Kb" << endl;
			goto Clear;
		}
		if (cudaSuccess != cudaMalloc((void **)&d_dev, size * sizeof(float))) {
			cout << "Can't allocate memory for diffusion coefficients: " << size * sizeof(float) / 1024 << " Kb" << endl;
			goto Clear;
		}
		if (
			cudaSuccess != cudaMalloc((void **)&tsa_dev, size * sizeof(float))
			|| cudaSuccess != cudaMalloc((void **)&tsb_dev, size * sizeof(float))
			|| cudaSuccess != cudaMalloc((void **)&tsc_dev, size * sizeof(float))
			|| cudaSuccess != cudaMalloc((void **)&tsd_dev, size * sizeof(float))
			|| cudaSuccess != cudaMalloc((void **)&tsx_dev, size * sizeof(float))
		) {
			cout << "Can't allocate memory for thomson sweep algorithm: " << 5 * size * sizeof(float) / 1024 << " Kb" << endl;
			goto Clear;
		}

		thomson_sweep_test_kernell<<<1,1>>>(f_dev,d_dev,tsa_dev,tsb_dev,tsc_dev,tsd_dev,tsx_dev,size,200000u);
		if (cudaSuccess != cudaGetLastError()) {
			cout << "Kernell launch failed: " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		else {
			cout << "Calculation Success!" << endl;
			if (cudaSuccess != cudaMemcpy(f_next.data(), f_dev, size * sizeof(float), cudaMemcpyDeviceToHost)) {
				cout << "Memory copy device->host failed!" << endl;
			}
			else {
				ofstream ascii_out("./data/f.txt");
				for (auto f : f_next) {
					ascii_out << f << '\n';
				}
			}
		}


	Clear:;
		if (f_dev != NULL) cudaFree(f_dev);
		if (d_dev != NULL) cudaFree(d_dev);
		if (tsa_dev != NULL) cudaFree(tsa_dev);
		if (tsb_dev != NULL) cudaFree(tsb_dev);
		if (tsc_dev != NULL) cudaFree(tsc_dev);
		if (tsd_dev != NULL) cudaFree(tsd_dev);
		if (tsx_dev != NULL) cudaFree(tsx_dev);
	}

	

	/*size_t size = 1000;
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
	}*/


	if (cudaSuccess != cudaDeviceReset()) {
		cout << "Error in device process termination!" << endl;
	}
End:;
	return 0;
}