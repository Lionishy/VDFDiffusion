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
		size_t x_size = 1024, y_size = 1024;
		vector<float> matrix(x_size * y_size), matrixT(x_size * y_size);
		init_matrix(matrix, x_size, y_size);

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
			constexpr unsigned const tile_dim = 32u, block_rows = 8u;
			auto begin = chrono::steady_clock::now(), end = begin;
			dim3 grid(x_size/tile_dim,y_size/tile_dim), threads(tile_dim, block_rows);
			for (unsigned rep = 0; rep != 10000; ++rep)
				transpose_kernell<tile_dim,block_rows><<<grid, threads>>>(mem_dev + x_size * y_size, mem_dev, x_size, y_size);
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
			ascii_out.setf(ios::scientific, ios::floatfield);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx) {
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					ascii_out << matrix[x_idx + y_idx * x_size] - matrixT[y_idx + x_idx*y_size] << " | ";
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