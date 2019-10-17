#include "Transpose.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

void init_matrix(std::vector<float> &matrix, size_t x_size, size_t y_size) {
	for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
		for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
			matrix[x_idx + y_idx * x_size] = float(x_idx) + 1.f / (float(y_idx)+1.f);
}

int main() {
	using namespace std;

	cudaError_t cudaStatus;
	float *mem_dev = NULL;

	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cout << "Error in starting cuda device:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}

	{
		size_t x_size = 1024, y_size = 1024, tile_size = 32;
		vector<float> matrix(x_size * y_size), matrixT(x_size*y_size);
		init_matrix(matrix, x_size, y_size);

		{
			ofstream ascii_out("./data/matrix.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx) {
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					ascii_out << matrix[x_idx + y_idx * x_size] << "|";
				ascii_out << "####" << endl;
			}
		}

		if (cudaSuccess != (cudaStatus = cudaMalloc((void **)&mem_dev, 2 * x_size * y_size * sizeof(float)))) {
			cout << "Can't allocate global memory of " << (x_size * y_size * sizeof(float)) / 1024 << " Kb" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}

		if (cudaSuccess != (cudaStatus = cudaMemcpy(mem_dev, matrix.data(), x_size * y_size * sizeof(float), cudaMemcpyHostToDevice))) {
			cout << "Can't copy data from host to device:" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}
		{
			auto begin = chrono::steady_clock::now(), end = begin;
			dim3 blockDim(x_size / tile_size, y_size / tile_size);
			dim3 threadDim(tile_size, tile_size);
			if (false) {
				for (unsigned rep = 0; rep != 1024; ++rep)
				simple_transpose_kernel<<<blockDim, threadDim>>>(mem_dev + x_size * y_size, mem_dev, x_size, y_size, tile_size);
			}

			if (false) {
				for (unsigned rep = 0; rep != 1024; ++rep)
				naiive_transpose_kernel<<<blockDim, threadDim>>>(mem_dev + x_size * y_size, mem_dev, x_size, y_size, tile_size);
			}

			if (true) {
				dim3 grid(32,32), threads(32, 8);
				for (unsigned rep = 0; rep != 10000; ++rep)
					paper_transpose_kernell <<<grid, threads>>>(mem_dev + x_size * y_size, mem_dev, x_size, y_size);
			}
			cudaDeviceSynchronize();

			end = chrono::steady_clock::now();
			cout << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
		}

		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Transpose kernel launch failed!" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}

		if (cudaSuccess != (cudaStatus = cudaMemcpy(matrixT.data(), mem_dev + x_size * y_size, x_size * y_size * sizeof(float), cudaMemcpyDeviceToHost))) {
			cout << "Can't copy data from device to host:" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto Clear;
		}

		{
			ofstream ascii_out("./data/matrixT.txt");
			ascii_out.precision(7); ascii_out.setf(ios::fixed, ios::floatfield);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx) {
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					ascii_out << matrix[x_idx + y_idx * x_size] - matrixT[y_idx + x_idx*y_size] << "|";
				ascii_out << "####" << endl;
			}
		}
	}

Clear:
	if (NULL != mem_dev) cudaFree(mem_dev);

	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cout << "Error in device process termination:" << endl;
		cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}

End:
	return 0;
}