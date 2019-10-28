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
#include <cmath>

template <typename T>
void initial_sin_wave(std::vector<T> &f, size_t x_size, size_t y_size, int N) {
	auto const PI = T(3.14159265358979323);
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
			f[x_idx * y_size + y_idx] = std::sin((PI * N) / (x_size-1)*x_idx);
}

template <typename T>
void initial_x_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
			dfc[y_idx + x_idx * y_size] = T(1);
}

template <typename T>
void initial_y_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		dfc[y_idx] = dfc[y_idx + (x_size - 2) * y_size] = T(0);

	for (size_t x_idx = 1; x_idx != x_size-2; ++x_idx)
		for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
			dfc[y_idx + x_idx * y_size] = T(0);
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

template <typename T>
cudaError_t iteration_step(T **f_prev, T **f_curr, T **f_tmp, T *x_dfc, T *y_dfc, T *a, T *b, T *c, T *d, T rx, T ry, size_t x_size, size_t y_size) {
	cudaError_t cudaStatus;
	int threads, blockDim;
	size_t matrix_shift = x_size + 1;

	blockDim = 1; threads = x_size - 2;
	iki::diffusion::device::forward_step_multisolver_kernel<<<blockDim, threads>>>(*f_prev, x_dfc, y_dfc, a, b, c, d, rx, ry, x_size - 2, y_size, x_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	iki::math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, *f_curr, x_size - 2, y_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	if (cudaSuccess != (cudaStatus = cycle_transpose<32u, 8u>(f_prev, f_curr, f_tmp, x_size, y_size, matrix_shift)))
		return cudaStatus;

	blockDim = 1; threads = y_size - 2;
	iki::diffusion::device::correction_step_multisolver_kernel<<<blockDim, threads>>>(*f_prev, *f_curr, y_dfc, a, b, c, d, ry, y_size - 2, x_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	iki::math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, *f_curr, y_size - 2, x_size);
	if (cudaSuccess != (cudaStatus = cudaGetLastError()))
		return cudaStatus;

	if (cudaSuccess != (cudaStatus = cycle_transpose<32u, 8u>(f_prev, f_curr, f_tmp, y_size, x_size, matrix_shift)))
		return cudaStatus;

	std::swap(*f_prev, *f_curr);
	return cudaStatus;
}

#include "TwoDimensionalDeviceSolver.cuh"

int main() {
	using namespace std;
	using namespace iki;

	cudaError_t cudaStatus;
	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}

	{
		size_t x_size = 1024, y_size = 1024;
		vector<float> f(x_size * y_size); initial_sin_wave(f, x_size, y_size, 2);
		vector<float> xx_dfc(x_size * y_size); initial_x_dfc(xx_dfc, x_size, y_size);
		vector<float> yy_dfc(x_size * y_size); initial_y_dfc(yy_dfc, y_size, x_size);

		try {
			iki::diffusion::TwoDimensionalSolver<float> solver(cerr, x_size, y_size, 1.f, 1.f, 1.f);
			solver.init(f.data(), xx_dfc.data(), yy_dfc.data(), yy_dfc.data(), yy_dfc.data());
			{
				auto begin = chrono::steady_clock::now(), end = begin;

				for (size_t count = 0; count != 10000; ++count)
					solver.step();

				cudaDeviceSynchronize();
				end = chrono::steady_clock::now();
				cerr << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
			}
			solver.retrieve(f.data());
		}
		catch (std::exception & ex) {
			cerr << ex.what() << endl;
		}

		{
			std::ofstream ascii_out("./data/matrix.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					ascii_out << x_idx << " " << y_idx << " " << f[x_idx * y_size + y_idx] << '\n';
		}
	}

End:
	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cerr << "Error in device process termination: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}
	return 0;
}