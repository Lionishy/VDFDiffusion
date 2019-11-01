#include "TwoDimensionalDeviceSolver.cuh"

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
void initial_sin_sin_wave(std::vector<T> &f, size_t x_size, size_t y_size, int Nx, int Ny) {
	auto const PI = T(3.14159265358979323);
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
			f[x_idx * y_size + y_idx] = std::sin((PI * Nx) / (x_size - 1) * x_idx) * std::sin((PI * Ny) / (y_size - 1) * y_idx);
}

template <typename T>
void initial_x_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
			dfc[y_idx + x_idx * y_size] = T(1);
}

template <typename T>
void initial_y_dfc(std::vector<T> &dfc, size_t x_size, size_t y_size) {
	for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
		for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
			dfc[y_idx + x_idx * y_size] = T(1);
}

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
		vector<double> f(x_size * y_size); initial_sin_sin_wave(f, x_size, y_size, 1, 1);
		vector<double> xx_dfc(x_size * y_size); initial_x_dfc(xx_dfc, x_size, y_size);
		vector<double> yy_dfc(x_size * y_size); initial_y_dfc(yy_dfc, y_size, x_size);
		vector<double> xy_dfc(x_size * y_size);

		try {
			iki::diffusion::TwoDimensionalSolver<double> solver(cerr, x_size, y_size, 0.7f, 0.7f, 1.f);
			solver.init(f.data(), xx_dfc.data(), yy_dfc.data(), xy_dfc.data(), xy_dfc.data());
			{
				auto begin = chrono::steady_clock::now(), end = begin;

				for (size_t count = 0; count != 30; ++count)
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