#include "SimpleTable.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <cmath>

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
struct PhysicalParameters {
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
PhysicalParameters<T> init_parameters(T nc, T betta_c, T TcTh_ratio, T bulk_to_alfven_c) {
	PhysicalParameters<T> p;
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

	__device__ ResonantVelocityEqn(T v_res, PhysicalParameters<T> params, ZFunc<T> Z): v_res(v_res), params(params), Z(Z) {  }

	T v_res;
	PhysicalParameters<T> params;
	ZFunc<T> Z;
};

template <typename T>
struct DispersionRootDerivative {
	__device__ T operator()(T omega, T k) const {
		T arg_c = (omega - T(1.)) / (k * p.betta_root_c) - p.bulk_to_term_c;
		T arg_h = (omega - T(1.)) / (k * p.betta_root_h) - p.bulk_to_term_h;
		T Zc = Z(arg_c), Zh = Z(arg_h);
		return p.nc / (k * p.betta_root_c) * (-Zc + (omega / (k * p.betta_root_c) - p.bulk_to_term_c) * (Zc * arg_c + T(1.)))
			+ p.nh / (k * p.betta_root_h) * (-Zh + (omega / (k * p.betta_root_h) - p.bulk_to_term_h) * (Zh * arg_h + T(1.)));
	}

	__device__ DispersionRootDerivative(PhysicalParameters<T> params, ZFunc<T> Z): params(params), Z(Z) { }

	PhysicalParameters<T> params;
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
__global__ void dispersion_relation_solve(T const *v_res, T *omega, T *derive, int *status, unsigned size, PhysicalParameters<T> params, T z_func_step, unsigned z_func_size, T *z_func_table) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	ResonantVelocityEqn<T> eqn(*(v_res + shift), params, ZFunc<T>(z_func_step, z_func_size, z_func_table));
	DispersionRootDerivative<T> dispersion_derive(params, ZFunc<T>(z_func_step, z_func_size, z_func_table));

	*(status + shift) = step_solver(eqn, T(0.), T(1.e-5), T(1. + 1.e-6), omega + shift);
	if (0 == *(status + shift)) {
		T k = (*(omega + shift) - T(1.)) / (*(v_res + shift) * params.betta_root_c);
		*(derive + shift) = dispersion_derive(*(omega + shift), k);
	}
}

template <int P, typename T>
T pow(T arg) {
	T res = T(1.);
	for (uint32_t counter = 0u; counter != P; ++counter)
		res *= arg;
	return res;
}



template <typename T>
struct VDF {
public:
	VDF(PhysicalParameters<T> params) : p(params) { }

	T operator()(T vperp, T vparall) const {
		T coeff_c = std::exp(-pow<2>(vperp) * T(0.5)), coeff_h = std::exp(-pow<2>(vperp) * T(0.5) * p.TcTh_ratio);
		return
			p.nc * coeff_c * std::exp(-T(0.5) * pow<2>(vparall - p.bulk_to_term_c))
			+ p.nh * pow<3>(std::sqrt(p.TcTh_ratio)) * coeff_h *
			std::exp(-T(0.5) * pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h));
	}

private:
	PhysicalParameters<T> p;
};

template <typename T>
class VDFUniformGridTabulator final {
public:
	VDFUniformGridTabulator(PhysicalParameters<T> params) : vdf(params) { }

	iki::UniformSimpleTable<T, 2u, 1u> operator()(iki::UniformSimpleTable<T, 2u, 1u> &table) {
		for (unsigned vparall_counter = 0u; vparall_counter != table.bounds.components[0]; ++vparall_counter) {
			for (unsigned vperp_counter = 0u; vperp_counter != table.bounds.components[1]; ++vperp_counter) {
				table.data[vperp_counter + vparall_counter * table.bounds.components[1]] =
					vdf(
						table.space.axes[1].begin + table.space.axes[1].step * vperp_counter
						, table.space.axes[1].begin + table.space.axes[1].step * vparall_counter
					);
			}
		}
		return table;
	}

private:
	VDF<T> vdf;
};

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

#include <chrono>

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
		return 0;
	}


	//CUDA
	unsigned size = 1024;
	unsigned bytes = size * (3 * sizeof(float) + sizeof(int)) + sizeof(float) * zfunc_table.bounds.components[0];
	void *global_memory = NULL;
	float *v_res_dev = NULL, *omegas_dev = NULL, *derive_dev = NULL, *zfunc_table_dev = NULL; int *status_dev = NULL;
	
	vector<float> v_res_data(size), omegas(size);
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
	derive_dev = omegas_dev + size;
	status_dev = (int *)(derive_dev + size);
	zfunc_table_dev = (float *)(status_dev + size);

	float start = -0.9f, step = -15.f / (size - 1);
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

	PhysicalParameters<float> params = init_parameters(0.85f, 1.f / 0.85f, 0.25f, -9.f);
	{
		auto begin = chrono::steady_clock::now(), end = begin;
		dispersion_relation_solve<<<1, 1024>>>(v_res_dev, omegas_dev, status_dev, size, params, zfunc_table.space.axes[0].step, zfunc_table.bounds.components[0], zfunc_table_dev);

		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Failed launch kernell!" << endl;
			cudaDeviceSynchronize();
			goto Clear;
		}

		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(omegas.data(), omegas_dev, size * sizeof(float), cudaMemcpyDeviceToHost))) {
		cerr << "Can't copy omega data from device to host: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	{
		ofstream ascii_os("./data/fdispersion.txt");
		ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
		for (unsigned idx = 0; idx != size; ++idx)
			ascii_os << v_res_data[idx] << " " << omegas[idx] << " " << (omegas[idx] - 1.f) / (v_res_data[idx] * params.betta_root_c) << '\n';
	}

Clear:
	if (NULL != global_memory) cudaFree(global_memory);
	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cerr << "Error in device process termination: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}

End:;
	return 0;
}