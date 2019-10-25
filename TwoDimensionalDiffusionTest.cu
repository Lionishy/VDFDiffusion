#include "ForwardStepKernel.cuh"
#include "CorrectionStepKernel.cuh"
#include "ThomsonSolverKernel.cuh"
#include "Transpose.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <utility>

template <typename T>
void initial_value(std::vector<T> &f, size_t x_size, size_t y_size) {
	T grad = T(1) / (y_size-1);
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx) {
			f[x_idx*y_size + y_idx] = T(1) - grad * y_idx;
		}
}

template <typename T>
void initial_x_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 1; y_idx != y_size-2; ++y_idx)
		for (size_t x_idx = 1; x_idx != x_size - 2; ++x_idx)
			dfc[y_idx + x_idx * y_size] = T(1);
}

template <typename T>
void initial_y_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		dfc[y_idx] = dfc[(x_size - 1) * y_size + y_idx] = T(0);

	for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
		dfc[x_idx * y_size] = dfc[y_size - 1 + x_idx * y_size] = T(0);

	for (size_t x_idx = 1; x_idx != x_size - 1; ++x_idx)
		for (size_t y_idx = 1; y_idx != y_size - 1; ++y_idx)
			dfc[x_idx * y_size + y_idx] = T(1);
}

template <unsigned tile_dim, unsigned block_rows, typename T>
cudaError_t cycle_transpose(T **f_prev, T **f_curr, T **f_tmp, size_t x_size, size_t y_size, size_t &matrix_shift) {
	cudaError_t cudaStatus;

	float *f_prev_full = *f_prev - matrix_shift, *f_curr_full = *f_curr - matrix_shift, *f_tmp_full = *f_tmp - matrix_shift;
	dim3 grid(x_size / tile_dim, y_size / tile_dim), threads(tile_dim, block_rows);
	iki::math::device::transpose_kernell<tile_dim, block_rows><<<grid, threads>>>(f_tmp_full, f_curr_full, x_size, y_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	iki::math::device::transpose_kernell<tile_dim, block_rows><<<grid, threads>>>(f_curr_full, f_prev_full, x_size, y_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	auto rotate_tmp = f_prev_full;
	f_prev_full = f_curr_full;
	f_curr_full = f_tmp_full;
	f_tmp_full = rotate_tmp;

	matrix_shift = x_size + 1;
	*f_prev = f_prev_full + matrix_shift;
	*f_curr = f_curr_full + matrix_shift;
	*f_tmp = f_tmp_full + matrix_shift;

	return cudaStatus;
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
	initial_value(f, x_size, y_size); initial_x_dfc(x_diffusion, x_size, y_size); initial_y_dfc(y_diffusion, x_size, y_size);
	float rx = 1.0f, ry = 1.0f;

	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}

	if (cudaSuccess != (cudaStatus = cudaMalloc((void **)&gm_dev, 9 * x_size * y_size * sizeof(float)))) {
		cerr << "Can't allocate global device memory of " << (9*x_size*y_size*sizeof(float)/1024) << " Kb: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}
	else {
		cerr << (9 * x_size * y_size * sizeof(float) / 1024) << " Kb: " << " successfully allocated!" << endl;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev, f.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cerr << "Can't copy data from f to device:" << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + x_size * y_size, f.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cerr << "Can't copy data from f to device:" << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + 3 * x_size * y_size, x_diffusion.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cout << "Can't copy data from x_dfc to device:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gm_dev + 4 * x_size * y_size, y_diffusion.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
		cerr << "Can't copy data from y_dfc to device:" << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	{
		size_t matrix_size = x_size * y_size, matrix_shift = y_size + 1;
		float *f_prev = gm_dev + y_size + 1, *f_curr = f_prev + matrix_size, *f_tmp = f_curr + matrix_size, *x_dfc = f_tmp + matrix_size, *y_dfc = x_dfc + matrix_size, *a = y_dfc + matrix_size, *b = a + matrix_size, *c = b + matrix_size, *d = c + matrix_size;

		int blockDim = 1, threads = x_size - 2;
		auto begin = chrono::steady_clock::now(), end = begin;
		for (int count = 0; count != 1000; ++count) {
			diffusion::device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, a, b, c, d, rx, ry, x_size - 2, y_size, x_size );
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cerr << "On iteration " << count << " forward step calculation kernel launch failed!" << endl;
				cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cerr << "On iteration " << count << " forward step thomson solver kernel launch failed!" << endl;
				cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}

			{
				if (cudaSuccess != (cudaStatus = cycle_transpose<32u,8u>(&f_prev,&f_curr,&f_tmp,x_size,y_size,matrix_shift))) {
					cerr << "On iteration " << count << " transposition after forward step failed!" << endl;
					cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
					cudaDeviceSynchronize();
					goto Clear;
				}
			}
			
			diffusion::device::correction_step_multisolver_kernel<<<blockDim,threads>>>(f_prev, f_curr, y_dfc, a, b, c, d, ry, y_size-2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cerr << "On iteration " << count << " correction step calculation kernel launch failed!" << endl;
				cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}

			math::device::thomson_multisolver_kernell<<<blockDim,threads>>>(a, b, c, d, f_curr, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
				cerr << "On iteration " << count << "correction step thomson solver kernel launch failed!" << endl;
				cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}

			{
				if (cudaSuccess != (cudaStatus = cycle_transpose<32u, 8u>(&f_prev, &f_curr, &f_tmp, y_size, x_size, matrix_shift))) {
					cerr << "On iteration " << count << " transposition after forward step failed!" << endl;
					cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
					cudaDeviceSynchronize();
					goto Clear;
				}
			}
			
			swap(f_prev, f_curr);
		}
		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;

		if (cudaSuccess != (cudaStatus = cudaMemcpy(f.data(), f_prev - matrix_shift, x_size*y_size * sizeof(float), cudaMemcpyDeviceToHost))) {
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