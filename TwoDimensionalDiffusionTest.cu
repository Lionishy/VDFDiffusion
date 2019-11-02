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
	vector<float> f(x_size * y_size), x_diffusion(x_size * y_size), y_diffusion(x_size * y_size);
	diffusion::x_y_sin_sin_test(f, x_diffusion, y_diffusion, x_size, y_size, 1, 1);

	float rx = 10.0f, ry = 10.0f;

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

		iki::diffusion::TwoDimensionalSolver<float> solver(x_size, y_size, rx, ry, f_prev, f_curr, f_tmp, x_dfc, y_dfc, a, b, c, d);

		auto begin = chrono::steady_clock::now(), end = begin;
		for (int count = 0; count != 1000; ++count) {
			if (cudaSuccess != (cudaStatus = solver.step())) {
				cerr << "On iteration " << count << " step kernell failed: " << endl;
				cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
				cudaDeviceSynchronize();
				goto Clear;
			}
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
			
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
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