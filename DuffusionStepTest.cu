#include "SimpleTwoDimensionalSolver.cuh"
#include "DiffusionTest.h"
#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>
#include <iostream>
#include <fstream>

int main() {
	using namespace std;
	using namespace iki;
	using namespace iki::diffusion;

	size_t const x_size = 1024, y_size = 1024, mtx_size = x_size * y_size;
	float rx = 1.0, ry = 1.0;

	UniformSimpleTable<float, 2u, 1u> vdf_table;
	vector<float> vdf_data;
	{
		vdf_table.bounds.components[0] = x_size;
		vdf_table.bounds.components[1] = y_size;
		//v parall x
		vdf_table.space.axes[0].begin = -15.0f;
		vdf_table.space.axes[0].step = 1.4e-2f;
		//v perp y
		vdf_table.space.axes[1].begin = 0.f;
		vdf_table.space.axes[1].step = 15.0f / (y_size - 1);

		vdf_data.resize(collapsed_size(&vdf_table.bounds));
		vdf_table.data = vdf_data.data();
	}

	vector<float> x_dfc(mtx_size), y_dfc(mtx_size), xy_dfc(mtx_size), yx_dfc(mtx_size);
	x_slope_test(vdf_data, x_dfc, y_dfc, x_size, y_size);


	{
		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
			cerr << "Error in starting cuda device: " << endl;
			cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			return -1;
		}
		SimpleTwoDimensionalSolver<float> diffusion_solver(x_size, y_size, rx, ry, vdf_data, x_dfc, y_dfc, xy_dfc, yx_dfc);

		if (cudaSuccess != (cudaStatus = cudaMemcpy(vdf_data.data(), diffusion_solver.f_curr_full, x_size * y_size * sizeof(float), cudaMemcpyDeviceToHost))) {
			cout << "Can't copy data from f_prev to host:" << endl;
			cout << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			return -1;
		}

		{
			ofstream ascii_os("./data/vdf.txt");
			ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
			ascii_os << vdf_table;
		}
	}

	return 0;
}