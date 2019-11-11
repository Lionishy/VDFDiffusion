#include "SimpleTable.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>

template <typename T>
struct ZFunc {
	__device__ T operator()(T arg) const {
		T farg = fabs(arg);
		auto idx = size_t(farg / step);
		if ((idx + 1u) < size) {
			return (arg > T(0.) ? T(1) : T(-1)) *((table[idx + 1u] - table[idx]) / step * (farg - step * idx) + table[idx]);
		}
		else { //asymptotic
			T over = T(1.) / arg, square = over * over;
			return -over * (T(1) + square + T(3) * square * square);
		}
	}

	__device__ ZFunc(T step, unsigned size, T *table): step(step), size(size), table(table) {  }

	T step;
	unsigned size;
	T *table;
};


template <typename T>
struct PhysicalParamenters {
	//fundamental parameters
	T nc;               //core particles density
	T TcTh_ratio;       //ratio of the core temperature to the halo temperature
	T betta_c;          //ratio of the core thermal pressure to the magnetic pressure
	T bulk_to_alfven_c; //bulk speed in terms of alfven speed

	//derived parameters
	T nh;
	T betta_root_c, betta_root_h;     //square root of the betta parameters core and halo
	T bulk_to_term_c, bulk_to_term_h; //bulk velocity in terms of thermal speed
};

template <typename T>
PhysicalParamenters<T> init_parameters(T nc, T betta_c, T TcTh_ratio, T bulk_to_alfven_c) {
	PhysicalParamenters<T> p;
	p.nc = nc;
	p.betta_c = betta_c;
	p.TcTh_ratio = TcTh_ratio;
	p.bulk_to_alfven_c = bulk_to_alfven_c;

	p.nh = T(1) - nc;
	p.betta_root_c = std::sqrt(T(0.5) * betta_c);
	p.betta_root_h = std::sqrt(T(0.5) * betta_c / TcTh_ratio);
	p.bulk_to_term_c = bulk_to_alfven_c / p.betta_root_c * std::sqrt(T(1. / 1836.));
	p.bulk_to_term_h = -(nc / p.nh) * p.bulk_to_term_c * std::sqrt(TcTh_ratio);

	return p;
}

template <typename T>
struct ResonantVelocityEqn {
	__device__ T operator()(T omega) const {
		T Zc = Z(v_res - params.bulk_to_term_c)
		, Zh = Z(v_res * sqrt(params.TcTh_ratio) - params.bulk_to_term_h);
		return T(1. / 1836.)
			+ (omega - T(1)) * (omega - T(1)) / (v_res * params.betta_root_c) / (v_res * params.betta_root_c)
			- params.nc * ((omega * v_res) / (omega - T(1)) - params.bulk_to_term_c) * Zc
			- params.nh * ((omega * v_res * sqrt(params.TcTh_ratio)) / (omega - T(1)) - params.bulk_to_term_h) * Zh;
	}

	__device__ ResonantVelocityEqn(T v_res, PhysicalParamenters<T> params, ZFunc<T> Z): v_res(v_res), params(params), Z(Z) {  }

	T v_res;
	PhysicalParamenters<T> params;
	ZFunc<T> Z;
};

template <typename T, typename Eqn_t>
__device__ int step_solver(Eqn_t f, T start, T step, T stop, T *res) {
	unsigned count = 0; T arg_curr = start + step * count, arg_next = start + step * (count + 1);
	T f_curr = f(arg_curr)
	, f_next = f(arg_next);
	while (arg_curr < stop) {
		if (f_curr * f_next < T(0)) { *res = T(0.5) * (arg_curr + arg_next); return 0; }
		
		arg_curr = arg_next;
		arg_next = start + step * ++count;
		f_curr = f_next;
		f_next = f(arg_next);
	}
	return 1;
}

template <typename T>
__global__ void dispersion_relation_solve(T const *v_res, T *omegas, int *status, unsigned size, PhysicalParamenters<T> params, T z_func_step, unsigned z_func_size, T *z_func_table) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	ResonantVelocityEqn<T> eqn(*(v_res + shift), params, ZFunc<T>(z_func_step, z_func_size, z_func_table));
	*(status + shift) = step_solver(eqn, T(0.), T(1.e-5), T(1. + 1.e-6), omegas + shift);
}

#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <iostream>
#include <fstream>
#include <vector>

template <typename T>
std::istream &ZFuncImport(std::istream &binary_is, iki::UniformSimpleTable<T, 1u, 1u> &zfunc_table, std::vector<T> &zfunc_data) {
	read_binary(binary_is, zfunc_table.space);
	read_binary(binary_is, zfunc_table.bounds);

	zfunc_data.resize(zfunc_table.bounds.components[0]);
	zfunc_table.data = zfunc_data.data();
	read_binary(binary_is, zfunc_table);
	return binary_is;
}

int main() {
	using namespace std;
	using namespace iki;

	/* Load ZFunc table */
	UniformSimpleTable<float, 1u, 1u> zfunc_table;
	vector<float> zfunc_data;
	try {
		std::ifstream binary_is("./data/fZFunc.tbl", ios::binary);
		binary_is.exceptions(ios::badbit | ios::failbit);
		ZFuncImport(binary_is, zfunc_table, zfunc_data);
	}
	catch (exception const &ex) {
		cerr << "Error in reading ZFunc data: " << ex.what() << endl;
	}

	//CUDA
	{
		unsigned size = 1024;
		unsigned bytes = size * (2 * sizeof(float) + sizeof(int)) + sizeof(float) * zfunc_table.bounds.components[0];
		void *global_memory = NULL;
		float *v_res_dev = NULL, *omegas_dev = NULL, *zfunc_table_dev; int *status_dev = NULL;

		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
			cerr << "Error in starting cuda device: " << endl;
			cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
			goto End;
		}
		
		if (cudaSuccess != (cudaStatus = cudaMalloc(&global_memory, bytes))) {
			cout << "Can't allocate memory!" << endl;
			goto Clear;
		}
		v_res_dev = (float *)global_memory;
		omegas_dev = v_res_dev + size;
		status_dev = (int *)(omegas_dev + size);
		zfunc_table_dev = (float *)(status_dev + size);

		{
			vector<float> v_res_data(size); float start = 0.9f, step = 15.f / (size-1);
			for (unsigned idx = 0; idx != size; ++idx)
				v_res_data[idx] = start + step * idx;
			
			if (cudaSuccess != (cudaStatus = cudaMemcpy(v_res_dev, v_res_data.data(), size * sizeof(float), cudaMemcpyHostToDevice))) {
				cout << "Can't copy v_res data from host to device!" << endl;
				goto Clear;
			}

			if (cudaSuccess != (cudaStatus = cudaMemcpy(zfunc_table_dev, zfunc_data.data(), sizeof(float) * zfunc_table.bounds.components[0], cudaMemcpyHostToDevice))) {
				cout << "Can't copy zfunc table data from host to device!" << endl;
				goto Clear;
			}
		}

		PhysicalParamenters<float> params = init_parameters(0.85f, 1.f / 0.85f, 0.25f, -11.f);
		dispersion_relation_solve<<<1, 1024>>>(v_res_dev, omegas_dev, status_dev, size, params, zfunc_table.space.axes[0].step, zfunc_table.bounds.components[0], zfunc_table_dev);

		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Failed launch kernell!" << endl;
		}
		else {
			cout << "Success!" << endl;
		}

	Clear:
		if (NULL != global_memory) cudaFree(global_memory);
		if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
			cerr << "Error in device process termination: " << endl;
			cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		}

	End:;
	}
	
	return 0;
}