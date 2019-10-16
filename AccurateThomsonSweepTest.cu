
#include "StridedThomsonSolver.cuh"

#include <cuda_runtime.h>

template <typename T>
__device__ void strided_set_initial_matrix(T *a, T *b, T *c, T *d, size_t size, size_t stride) {
	a[0] = T(0); b[0] = T(3); c[0] = T(1); d[0] = T(4);
	for (size_t idx = 1; idx != size - 1; ++idx) {
		size_t stride_idx = stride * idx;
		a[stride_idx] = T(1);
		b[stride_idx] = T(2);
		c[stride_idx] = T(1);
		d[stride_idx] = T(4);
	}

	size_t last_idx = stride * (size - 1);
	a[last_idx] = T(1);
	b[last_idx] = T(3);
	c[last_idx] = T(0);
	d[last_idx] = T(4);
}

template <typename T>
__global__ void strided_accurate_thomson_sweep_test_kernell(T *mem, size_t size, size_t span, size_t loop_count) {
	size_t shift = threadIdx.x + blockDim.x * blockIdx.x;
	size_t grid_size = size * span, stride = span;
	T *f = mem, *dfc = f + grid_size, *a = dfc + grid_size, *b = a + grid_size, *c = b + grid_size, *d = c + grid_size;
	strided_set_initial_matrix(a + shift, b + shift, c + shift, d + shift, size, stride);
	iki::math::device::accurate_strided_thomson_sweep(a + shift, b + shift, c + shift, d + shift, f + shift, size, stride);


	/*size_t shift = threadIdx.x + blockDim.x * blockIdx.x;
	size_t grid_size = size * span, stride = span;
	T *f = mem, *dfc = f + grid_size, *a = dfc + grid_size, *b = a + grid_size, *c = b + grid_size, *d = c + grid_size;
	strided_set_initial_state(f + shift, dfc + shift, size, stride);
	for (; loop_count != 0; --loop_count)
		iki::diffusion::device::strided_diffusion_step(f + shift, dfc + shift, a + shift, b + shift, c + shift, d + shift, T(1.), size, stride);*/
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
		size_t size = 1024, span = 1024;

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
			unsigned threads_count = 512, blocks_count = span / threads_count;
			auto begin = chrono::steady_clock::now(), end = begin;
			strided_accurate_thomson_sweep_test_kernell <<<1, 1>>> (mem_dev, size, span, 1u);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cout << "Kernell launch failed: " << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				goto Clear;
			}
			cudaDeviceSynchronize();
			end = chrono::steady_clock::now();
			cout << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
		}
		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Kernell execution failed: " << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		else {
			cout << "Calculation Success!" << endl;
			vector<float> f_next(size * span); //to temporary save data from GPU for the further export into .txt file
			ofstream ascii_out("./data/f.txt"); ascii_out.precision(7); ascii_out.setf(std::ios::fixed, std::ios::floatfield);
			if (cudaSuccess != cudaMemcpy(f_next.data(), mem_dev, span * size * sizeof(float), cudaMemcpyDeviceToHost)) {
				cout << "Memory copy device->host failed!" << endl;
			}
			else {
				for (size_t row_idx = 0; row_idx != span; ++row_idx)
					for (size_t idx = 0; idx != size; ++idx)
						ascii_out << row_idx << ' ' << idx << ' ' << f_next[row_idx + idx*span] << '\n';
				ascii_out << endl;
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