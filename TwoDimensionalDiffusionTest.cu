#include "DiffusionTest.h"
#include "TwoDimensionalSolver.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <utility>
#include <cmath>

int main() {
	using namespace std;
	using namespace iki;

	cudaError_t cudaStatus;
	float *gm_dev = NULL;
	size_t x_size = 512, y_size = 512;
	vector<float> f(x_size * y_size), x_diffusion(x_size * y_size), y_diffusion(x_size * y_size), xy_diffusion(x_size * y_size), yx_diffusion(x_size * y_size);
	diffusion::x_y_sin_sin_mixed_term_test(f, x_diffusion, y_diffusion, xy_diffusion, yx_diffusion, x_size, y_size, 1, 1);

	float rx = 10.0f, ry = 10.0f, rxy = std::sqrt(rx * ry);

	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}

	{
		iki::diffusion::TwoDimensionalSolver<float> solver(x_size, y_size, rx, ry, f, x_diffusion, y_diffusion, xy_diffusion, yx_diffusion);
		auto begin = chrono::steady_clock::now(), end = begin;
		for (int count = 0; count != 10; ++count)
			solver.step();
		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;

		if (cudaSuccess != (cudaStatus = cudaMemcpy(f.data(), solver.f_prev_full, x_size*y_size * sizeof(float), cudaMemcpyDeviceToHost))) {
			cout << "Can't copy data from f_prev to host:" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		else {
			ofstream ascii_out("./data/matrix.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
					ascii_out << x_idx << " " << y_idx << " " << f[x_idx * y_size + y_idx] << endl;
		}
	}

Clear:
	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cerr << "Error in device process termination: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}

End:
	return 0;
}