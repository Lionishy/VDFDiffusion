#include "SimpleTable.h"
#include "SimpleTableIO.h"
#include "PhysicalParameters.h"
#include "ZFunc.cuh"
#include "ZFuncImport.h"
#include "Pow.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <cmath>

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

	__device__ ResonantVelocityEqn(T v_res, iki::whfi::PhysicalParameters<T> params, iki::ZFunc<T> Z): v_res(v_res), params(params), Z(Z) {  }

	T v_res;
	iki::whfi::PhysicalParameters<T> params;
	iki::ZFunc<T> Z;
};

template <typename T>
struct DispersionRootDerivative {
	__device__ T operator()(T omega, T k) const {
		T arg_c = (omega - T(1.)) / (k * params.betta_root_c) - params.bulk_to_term_c;
		T arg_h = (omega - T(1.)) / (k * params.betta_root_h) - params.bulk_to_term_h;
		T Zc = Z(arg_c), Zh = Z(arg_h);
		return params.nc / (k * params.betta_root_c) * (-Zc + (omega / (k * params.betta_root_c) - params.bulk_to_term_c) * (Zc * arg_c + T(1.)))
			+ params.nh / (k * params.betta_root_h) * (-Zh + (omega / (k * params.betta_root_h) - params.bulk_to_term_h) * (Zh * arg_h + T(1.)));
	}

	__device__ DispersionRootDerivative(iki::whfi::PhysicalParameters<T> params, iki::ZFunc<T> Z): params(params), Z(Z) { }

	iki::whfi::PhysicalParameters<T> params;
	iki::ZFunc<T> Z;
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
__global__ void dispersion_relation_solve(T const *v_res, T *omega, T *derive, int *status, unsigned size, iki::whfi::PhysicalParameters<T> params, T z_func_step, unsigned z_func_size, T *z_func_table) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	ResonantVelocityEqn<T> eqn(*(v_res + shift), params, iki::ZFunc<T>(z_func_step, z_func_size, z_func_table));
	DispersionRootDerivative<T> dispersion_derive(params, iki::ZFunc<T>(z_func_step, z_func_size, z_func_table));

	*(status + shift) = step_solver(eqn, T(0.), T(1.e-5), T(1. + 1.e-6), omega + shift);
	if (0 == *(status + shift)) {
		T k = (*(omega + shift) - T(1.)) / (*(v_res + shift) * params.betta_root_c);
		*(derive + shift) = dispersion_derive(*(omega + shift), k);
	}
}

template <typename T>
struct VDF {
public:
	VDF(iki::whfi::PhysicalParameters<T> params) : p(params) { }

	T operator()(T vperp, T vparall) const {
		T coeff_c = std::exp(-iki::math::pow<2>(vperp) * T(0.5)), coeff_h = std::exp(-pow<2>(vperp) * T(0.5) * p.TcTh_ratio);
		return
			p.nc * coeff_c * std::exp(-T(0.5) * iki::math::pow<2>(vparall - p.bulk_to_term_c))
			+ p.nh * iki::math::pow<3>(std::sqrt(p.TcTh_ratio)) * coeff_h *
			std::exp(-T(0.5) * iki::math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h));
	}

private:
	iki::whfi::PhysicalParameters<T> p;
};

template <typename T>
struct VDFmu {
public:
	VDFmu(iki::whfi::PhysicalParameters<T> params) : p(params) { }

	T operator()(T mu, T vparall) const {
		T coeff_c = std::exp(-mu), coeff_h = std::exp(-mu * p.TcTh_ratio);
		return
			p.nc * coeff_c * std::exp(-T(0.5) * iki::math::pow<2>(vparall - p.bulk_to_term_c))
			+ p.nh * iki::math::pow<3>(std::sqrt(p.TcTh_ratio)) * coeff_h *
			std::exp(-T(0.5) * iki::math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h));
	}

private:
	iki::whfi::PhysicalParameters<T> p;
};

template <typename T>
class VDFUniformGridTabulator final {
public:
	VDFUniformGridTabulator(iki::whfi::PhysicalParameters<T> params) : vdf(params) { }

	//sizes: 0 - vparall, 1 - vperp
	iki::UniformSimpleTable<T, 2u, 1u> vperp_near(iki::UniformSimpleTable<T, 2u, 1u> &table) {
		for (unsigned vparall_counter = 0u; vparall_counter != table.bounds.components[0]; ++vparall_counter) {
			for (unsigned vperp_counter = 0u; vperp_counter != table.bounds.components[1]; ++vperp_counter) {
				table.data[vperp_counter + vparall_counter * table.bounds.components[1]] =
					vdf(
						table.space.axes[1].begin + table.space.axes[1].step * vperp_counter
						, table.space.axes[0].begin + table.space.axes[0].step * vparall_counter
					);
			}
		}
		return table;
	}

	//sizes: 0 - vperp, 1 - vparall
	iki::UniformSimpleTable<T, 2u, 1u> vparall_near(iki::UniformSimpleTable<T, 2u, 1u> &table) {
		for (unsigned vperp_counter = 0u; vperp_counter != table.bounds.components[0]; ++vperp_counter) {
			for (unsigned vparall_counter = 0u; vparall_counter != table.bounds.components[1]; ++vparall_counter) {
				table.data[vparall_counter + vperp_counter * table.bounds.components[1]] = 
					vdf(
						table.space.axes[0].begin + table.space.axes[0].step * vperp_counter
						, table.space.axes[1].begin + table.space.axes[1].step * vparall_counter
					);
			}
		}
		return table;
	}

private:
	VDFmu<T> vdf;
};

#include "ZeroMoment.cuh"
#include "FirstMoment.cuh"

template <typename T>
__global__ void zero_moment_kernel(T const *f, T start, T dx, unsigned x_size, unsigned y_size, T *zero_moment) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	T s, rem;
	iki::math::device::zero_moment(f + shift + y_size, start, dx, x_size-2, y_size, &s, &rem);
	s -= T(0.5) * (*(f + shift) + *(f + shift + y_size * (x_size - 1)));
	*(zero_moment + shift) = s * dx;
}

template <typename T>
__global__ void first_moment_kernel(T const *f, T start, T dx, unsigned x_size, unsigned y_size, T *first_moment) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	T s, rem;
	iki::math::device::first_moment(f + shift + y_size, start, dx, x_size - 2, y_size, &s, &rem);
	s -= T(0.5) * (*(f + shift) * start + *(f + shift + y_size * (x_size - 1)) * (start + dx * (x_size - 1)));
	*(first_moment + shift) = s * dx;
}

template <typename T>
__global__ void gamma_kernel(T const *zero_moment, T const *first_moment, T const *omega, T const *dispersion_derive, T vparall_begin, T vparall_step, unsigned size, T *gamma) {
	unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
	if (0 == shift || size - 1 == shift) {
		*(gamma + shift) = 0; return;
	}
	T k_betta = (*(omega + shift) - T(1)) / (vparall_begin + vparall_step * shift);
	T first_moment_derive = T(0.5) * (first_moment[shift + 1] - first_moment[shift - 1]) / vparall_step;
	gamma[shift] = - T(1.25331414) / k_betta * (k_betta * first_moment_derive - zero_moment[shift]) / dispersion_derive[shift];
}

#include <iostream>
#include <fstream>
#include <vector>



template <typename T>
class AnalyticalMoments final {
public:
	AnalyticalMoments(iki::whfi::PhysicalParameters<T> p) : p(p) { }

	std::vector<T> g(T vparall_begin, T vparall_step, unsigned size) const {
		auto g_vparall = [this] (T vparall) {
			return p.nc * std::exp(-pow<2>(vparall - p.bulk_to_term_c) / T(2.)) + p.nh * std::sqrt(p.TcTh_ratio) * exp(-pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2.));
		};
		std::vector<T> table(size);
		for (unsigned count = 0u; count < size; ++count)
			table[count] = g_vparall(vparall_begin + vparall_step * count);
		return table;
	}

	std::vector<T> G(T vparall_begin, T vparall_step, unsigned size) const {
		auto G_vparall = [this] (T vparall) {
			return p.nc * std::exp(-pow<2>(vparall - p.bulk_to_term_c) / T(2.)) + p.nh * std::sqrt(T(1.) / p.TcTh_ratio) * exp(-pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2.));
		};
		std::vector<T> table(size);
		for (unsigned count = 0u; count < size; ++count)
			table[count] = G_vparall(vparall_begin + vparall_step * count);
		return table;
	}

private:
	iki::whfi::PhysicalParameters<T> p;
};

#include <chrono>

int main() {
	using namespace std;
	using namespace iki;

	whfi::PhysicalParameters<float> params = whfi::init_parameters(0.85f, 1.f / 0.85f, 0.25f, -9.f);

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
	
	unsigned size = 1024;
	UniformSimpleTable<float, 2u, 1u> vdf_table; //0 - vperp, 1 - vparall
	{
		vdf_table.bounds.components[0] = vdf_table.bounds.components[1] = size;
		vdf_table.space.axes[0].begin = 0.f;
		vdf_table.space.axes[0].step = 100.f / (size - 1);
		vdf_table.space.axes[1].begin = -1.0f;
		vdf_table.space.axes[1].step = -15.f / (size - 1);
	}
	vector<float> vdf_data(collapsed_size(&vdf_table.bounds));
	{
		vdf_table.data = vdf_data.data();
	}
	VDFUniformGridTabulator<float>(params).vparall_near(vdf_table);

	vector<float> v_res_data(size), omegas(size), zero_moment_host(size), first_moment_host(size), gamma(size);
	for (unsigned idx = 0; idx != size; ++idx)
		v_res_data[idx] = vdf_table.space.axes[1].begin + vdf_table.space.axes[1].step * idx;

	//CUDA
	unsigned bytes = size * size * sizeof(float) + size * (6 * sizeof(float) + sizeof(int)) + sizeof(float) * zfunc_table.bounds.components[0];
	void *global_memory = NULL;
	float *f = NULL, *v_res_dev = NULL, *omegas_dev = NULL, *derive_dev = NULL, *zero_moment = NULL, *first_moment = NULL, *gamma_dev = NULL, *zfunc_table_dev = NULL; int *status_dev = NULL;
	
	
	cudaError_t cudaStatus;
	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		goto End;
	}
		
	if (cudaSuccess != (cudaStatus = cudaMalloc(&global_memory, bytes))) {
		cerr << "Can't allocate memory!" << endl;
		goto Clear;
	}
	else {
		cerr << "Successfully allocated " << ((bytes - 1) / 1024 + 1) << " Kb on device!" << endl;
	}
	f = (float *)global_memory;
	v_res_dev = f + size * size;
	omegas_dev = v_res_dev + size;
	derive_dev = omegas_dev + size;
	zero_moment = derive_dev + size;
	first_moment = zero_moment + size;
	gamma_dev = first_moment + size;
	status_dev = (int *)(gamma_dev + size);
	zfunc_table_dev = (float *)(status_dev + size);

	if (cudaSuccess != (cudaStatus = cudaMemcpy(f, vdf_data.data(), size * size * sizeof(float), cudaMemcpyHostToDevice))) {
		cerr << "Can't copy f data from host to device!" << endl;
		goto Clear;
	}
			
	if (cudaSuccess != (cudaStatus = cudaMemcpy(v_res_dev, v_res_data.data(), size * sizeof(float), cudaMemcpyHostToDevice))) {
		cerr << "Can't copy v_res data from host to device!" << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(zfunc_table_dev, zfunc_data.data(), sizeof(float) * zfunc_table.bounds.components[0], cudaMemcpyHostToDevice))) {
		cerr << "Can't copy zfunc table data from host to device!" << endl;
		goto Clear;
	}

	
	{
		auto begin = chrono::steady_clock::now(), end = begin;
		dispersion_relation_solve<<<1, size>>>(v_res_dev, omegas_dev, derive_dev, status_dev, size, params, zfunc_table.space.axes[0].step, zfunc_table.bounds.components[0], zfunc_table_dev);

		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Failed launch kernell!" << endl;
			cudaDeviceSynchronize();
			goto Clear;
		}

		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Solving time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
	}

	{
		auto begin = chrono::steady_clock::now(), end = begin;
		zero_moment_kernel<<<1, size>>>(f, vdf_table.space.axes[0].begin, vdf_table.space.axes[0].step, size, size, zero_moment);

		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Failed launch zero moment kernell!" << endl;
			cudaDeviceSynchronize();
			goto Clear;
		}

		first_moment_kernel<<<1, size>>>(f, vdf_table.space.axes[0].begin, vdf_table.space.axes[0].step, size, size, first_moment);
		if (cudaSuccess != (cudaStatus = cudaGetLastError())) {
			cout << "Failed launch first moment kernell!" << endl;
			cudaDeviceSynchronize();
			goto Clear;
		}

		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Solving time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
	}

	{
		auto begin = chrono::steady_clock::now(), end = begin;
		gamma_kernel<<<1, size>>>(zero_moment, first_moment, omegas_dev, derive_dev, vdf_table.space.axes[1].begin, vdf_table.space.axes[1].step, size, gamma_dev);

		cudaDeviceSynchronize();
		end = chrono::steady_clock::now();
		cerr << "Solving time consumed: " << chrono::duration <double, milli>(end - begin).count() << " ms" << endl;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(omegas.data(), omegas_dev, size * sizeof(float), cudaMemcpyDeviceToHost))) {
		cerr << "Can't copy omega data from device to host: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	
	if (cudaSuccess != (cudaStatus = cudaMemcpy(zero_moment_host.data(), zero_moment, size * sizeof(float), cudaMemcpyDeviceToHost))) {
		cerr << "Can't copy zero moment data from device to host: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(first_moment_host.data(), first_moment, size * sizeof(float), cudaMemcpyDeviceToHost))) {
		cerr << "Can't copy first moment data from device to host: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	if (cudaSuccess != (cudaStatus = cudaMemcpy(gamma.data(), gamma_dev, size * sizeof(float), cudaMemcpyDeviceToHost))) {
		cerr << "Can't copy gamma data from device to host: " << cudaGetErrorString(cudaStatus) << endl;
		goto Clear;
	}

	{
		ofstream ascii_os("./data/fdispersion.txt");
		ascii_os.precision(7); ascii_os.setf(ios::fixed, ios::floatfield);
		for (unsigned idx = 0; idx != size; ++idx)
			ascii_os << v_res_data[idx] << " " << omegas[idx] << " " << (omegas[idx] - 1.f) / (v_res_data[idx] * params.betta_root_c) << " " << zero_moment_host[idx] << " " << first_moment_host[idx] << " " << gamma[idx] <<  '\n';
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