
#include "PhysicalParameters.h"
#include "ZFunc.h"
#include "ZFuncImport.h"
#include "SimpleTable.h"
#include "SimpleTableIO.h"
#include "InitialConditionsClass.h"
#include "GammaRecalculationClass.cuh"



#include <iostream>
#include <fstream>
#include <algorithm>


#include "Pow.h"
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

int main() {
	using namespace std;
	using namespace iki;

	whfi::PhysicalParameters<float> params = whfi::init_parameters(0.85f, 1.f / 0.85f, 0.25f, -9.f);
	UniformSpace<float, 2u> v_space;
	v_space.axes[0].begin = 0.f;
	v_space.axes[0].step = 5e-2f;
	v_space.axes[1].begin = -15.0f;
	v_space.axes[1].step = 1.3e-2f;

	// Load ZFunc table
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

	whfi::InitialConditionsCalculation<float> initial_cond_calc(ZFunc<float>(zfunc_table.space.axes[0].step,zfunc_table.bounds.components[0],zfunc_table.data));

	auto result = initial_cond_calc.recalculate(params,-15.f,1.3e-2f,1024);
	vector<float> x_dfc_pivot(1024 * 1024), y_dfc_pivot(1024 * 1024), xy_dfc_pivot(1024 * 1024), yx_dfc_pivot(1024 * 1024);
	{
		for (unsigned perp_count = 0; perp_count != 1024; ++perp_count) {
			float v_perp = v_space.axes[0].begin + v_space.axes[0].step * perp_count;
			for (unsigned parall_count = 0; parall_count != 1024; ++parall_count) {
				x_dfc_pivot[parall_count + perp_count * 1024] = result.Dpr[parall_count] ;
				y_dfc_pivot[parall_count + perp_count * 1024] = result.Dpl[parall_count];
				xy_dfc_pivot[parall_count + perp_count * 1024] = result.Dmx[parall_count] ;
				yx_dfc_pivot[parall_count + perp_count * 1024] = result.Dmx[parall_count] ;
			}
		}
			
	}

	cudaError_t cudaStatus;
	if (cudaSuccess != (cudaStatus = cudaSetDevice(0))) {
		cerr << "Error in starting cuda device: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
		return -1;
	}
	try {
		DeviceMemory external_memory(1024 * 1024 * 5 * sizeof(float));
		float *vdf = (float *)external_memory.get();
		float *x_dfc = vdf + 1024 * 1024;
		float *y_dfc = x_dfc + 1024 * 1024;
		float *xy_dfc = y_dfc + 1024 + 1024;
		float *yx_dfc = xy_dfc + 1024 + 1024;

		vector<float> amplitude_spectrum(1024, 1.e-5f);
		whfi::GammaRecalculation<float> gamma_recalc(1024, 1024, v_space, 1., x_dfc_pivot, y_dfc_pivot, xy_dfc_pivot, yx_dfc_pivot, result.dispersion_derivative, result.k_betta, amplitude_spectrum);

		gamma_recalc.external_memory_init(vdf, x_dfc, y_dfc, xy_dfc, yx_dfc);
		{
			UniformSimpleTable<float, 2u, 1u> vdf_table;
			vdf_table.space = v_space;
			vdf_table.bounds.components[0] = 1024;
			vdf_table.bounds.components[1] = 1024;
			
			vector<float> vdf_data(1024 * 1024);
			vdf_table.data = vdf_data.data();

			VDFUniformGridTabulator<float>(params).vparall_near(vdf_table);
			if (cudaSuccess != (cudaStatus = cudaMemcpy(vdf, vdf_table.data, 1024 * 1024 * sizeof(float), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);

			/*{
				ofstream ascii_os("./data/new_vdf.txt");
				ascii_os.precision(7); ascii_os.setf(ios::floatfield, ios::fixed);
				ascii_os << vdf_table;
			}*/
		}
		gamma_recalc.growth_rate_update();

		UniformSimpleTable<float, 1u, 2u> new_gamma_table;
		new_gamma_table.space.axes[0] = v_space.axes[1];
		new_gamma_table.bounds.components[0] = 1024;

		vector<float> new_gamma(1024), new_k_betta(1024);

		if (cudaSuccess != (cudaStatus = cudaMemcpy(new_gamma.data(), gamma_recalc.growth_rate_spectrum, 1024 * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);
		if (cudaSuccess != (cudaStatus = cudaMemcpy(new_k_betta.data(), gamma_recalc.k_betta, 1024 * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

		vector<float> new_gamma_table_data(1024 * 2);
		for (unsigned count = 0; count != 1024; ++count) {
			new_gamma_table_data[2 * count] = new_gamma[count];
			new_gamma_table_data[2 * count + 1] = new_k_betta[count] / params.betta_root_c;
		}

		new_gamma_table.data = new_gamma_table_data.data();
		{
			ofstream ascii_os("./data/new_gamma.txt");
			ascii_os.precision(7); ascii_os.setf(ios::floatfield, ios::fixed);
			ascii_os << new_gamma_table;
		}

		if (cudaSuccess != (cudaStatus = cudaMemcpy(x_dfc_pivot.data(), gamma_recalc.x_dfc, 1024 * 1024 * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);
		if (cudaSuccess != (cudaStatus = cudaMemcpy(y_dfc_pivot.data(), gamma_recalc.y_dfc, 1024 * 1024 * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);
		if (cudaSuccess != (cudaStatus = cudaMemcpy(xy_dfc_pivot.data(), gamma_recalc.xy_dfc, 1024 * 1024 * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

		UniformSimpleTable<float, 2u, 3u> dfc_table;
		vector<float> dfc_data(1024 * 1024 * 3);
		dfc_table.space = v_space;
		dfc_table.bounds.components[0] = 1024;
		dfc_table.bounds.components[1] = 1024;
		dfc_table.data = dfc_data.data();

		for (unsigned perp_count = 0; perp_count != 1024; ++perp_count) {
			for (unsigned parall_count = 0; parall_count != 1024; ++parall_count) {
				dfc_data[3 * (parall_count + perp_count * 1024)] = x_dfc_pivot[parall_count + perp_count * 1024];
				dfc_data[3 * (parall_count + perp_count * 1024) + 1] = y_dfc_pivot[parall_count + perp_count * 1024];
				dfc_data[3 * (parall_count + perp_count * 1024) + 2] = xy_dfc_pivot[parall_count + perp_count * 1024];
			}
		}

		{
			ofstream ascii_os("./data/new_dfc.txt");
			ascii_os.exceptions(ios::badbit | ios::failbit);
			ascii_os.precision(7); ascii_os.setf(ios::floatfield, ios::fixed);
			ascii_os << dfc_table;
		}


	}
	catch (std::exception const &ex) {
		cout << ex.what() << endl;
	}
	catch (...) {
		cout << "Unknown exception..." << endl;
	}

	if (cudaSuccess != (cudaStatus = cudaDeviceReset())) {
		cerr << "Error in device process termination: " << endl;
		cerr << cudaStatus << " -- " << cudaGetErrorString(cudaStatus) << endl;
	}

	return 0;
}