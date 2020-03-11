#include "SimpleTwoDimensionalSolver.cuh"
#include "SimpleTable.h"
#include "PhysicalParameters.h"
#include "VDFmu.h"
#include "GammaRecalculationClass.cuh"

#include "Pow.h"

#include <vector>


#include "InitialConditionsClass.h"
#include "ZFuncImport.h"
#include <iostream>
#include <fstream>
auto initial_conditions(iki::whfi::PhysicalParameters<float> params, iki::UniformAxis<float> vparall_axis, size_t vparall_size) {
	using namespace std;
	using namespace iki;
	// Load ZFunc table
	UniformSimpleTable<float, 1u, 1u> zfunc_table;
	vector<float> zfunc_data;
	ifstream binary_is("./data/fZFunc.tbl", ios::binary);
	binary_is.exceptions(ios::badbit | ios::failbit);
	ZFuncImport(binary_is, zfunc_table, zfunc_data);

	return whfi::InitialConditionsCalculation<float>(ZFunc<float>(zfunc_table.space.axes[0].step, zfunc_table.bounds.components[0], zfunc_table.data)).recalculate(params, vparall_axis.begin, vparall_axis.step, vparall_size);
}

void dfc_pivot_recalc(iki::UniformSimpleTable<float, 2u, 1u> const &vdf_table, std::vector<float> const &Dpl, std::vector<float> const &Dmx, std::vector<float> const &Dpr, std::vector<float> &x_dfc, std::vector<float> &y_dfc, std::vector<float> &xy_dfc, std::vector<float> &yx_dfc) {
	for (unsigned perp_count = 0; perp_count != vdf_table.bounds.components[1]; ++perp_count) {
		float v_perp = vdf_table.space.axes[1].begin + vdf_table.space.axes[1].step * perp_count;
		for (unsigned parall_count = 0; parall_count != vdf_table.bounds.components[0]; ++parall_count) {
			x_dfc[parall_count + perp_count * vdf_table.bounds.components[0]] = 1.e-4 * Dpr[parall_count] * v_perp;
			xy_dfc[parall_count + perp_count * vdf_table.bounds.components[0]] = 1.e-4 * Dmx[parall_count] * v_perp;

		}
	}

	for (unsigned parall_count = 0; parall_count != vdf_table.bounds.components[0]; ++parall_count) {
		float const Dpl_ = Dpl[parall_count], Dmx_ = Dmx[parall_count];
		for (unsigned perp_count = 0; perp_count != vdf_table.bounds.components[1]; ++perp_count) {
			float v_perp = vdf_table.space.axes[1].begin + vdf_table.space.axes[1].step * perp_count;
			y_dfc[parall_count * vdf_table.bounds.components[0] + perp_count] = 1.e-4 * Dpl_ * v_perp;
			yx_dfc[parall_count * vdf_table.bounds.components[0] + perp_count] = 1.e-4 * Dmx_ * v_perp;
		}
	}
}

int main() {
	using namespace std;
	using namespace iki;
	using namespace iki::diffusion;
	using namespace iki::whfi;

	PhysicalParameters<float> params = init_parameters(0.85f, 1.f / 0.85f, 0.25f, -9.f);;

	UniformSimpleTable<float, 2u, 1u> vdf_table;
	vector<float> vdf_vector;
	{ //initialize vdf_table
		vdf_table.bounds.components[0] = vdf_table.bounds.components[1] = 1024;
		vdf_table.space.axes[0].begin = -15.0f; //vparall
		vdf_table.space.axes[0].step = 1.3e-2f;
		vdf_table.space.axes[1].begin = 0.f;    //vperp
		vdf_table.space.axes[1].step = 3e-2f;
		vdf_vector.resize(collapsed_size(&vdf_table.bounds));
		vdf_table.data = vdf_vector.data();
	}

	float dt = 0.01; //dt = 1./omega_c
	float rparall = dt / math::pow<2u>(vdf_table.space.axes[0].step), rperp = dt / math::pow<2u>(vdf_table.space.axes[1].step);

	VDFmuUniformGridTabulator<float>(params).vparall_near(vdf_table);
	try { 

		{
			ofstream ascii_os("./data/vdf_init.txt");
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
			ascii_os << vdf_table;
		}


		auto result = initial_conditions(params, vdf_table.space.axes[0], vdf_table.bounds.components[0]);
		vector<float> x_dfc_pivot(collapsed_size(&vdf_table.bounds)), y_dfc_pivot(collapsed_size(&vdf_table.bounds)), xy_dfc_pivot(collapsed_size(&vdf_table.bounds)), yx_dfc_pivot(collapsed_size(&vdf_table.bounds)), amplitude_spectrum(vdf_table.bounds.components[0],1.e-4);
		dfc_pivot_recalc(vdf_table, result.Dpl, result.Dmx, result.Dpr, x_dfc_pivot, y_dfc_pivot, xy_dfc_pivot, yx_dfc_pivot);

		cudaError_t cudaStatus;
		if (cudaSuccess != (cudaStatus = cudaSetDevice(0)))
			throw runtime_error(cudaGetErrorString(cudaStatus));

		SimpleTwoDimensionalSolver<float> vdf_diffusor(vdf_table.bounds.components[0], vdf_table.bounds.components[1], rparall, rperp, vdf_vector, x_dfc_pivot, y_dfc_pivot, xy_dfc_pivot, yx_dfc_pivot);
		GammaRecalculation<float> growthrate(vdf_table.bounds.components[1],vdf_table.bounds.components[0],vdf_table.space,dt,x_dfc_pivot,y_dfc_pivot,xy_dfc_pivot,yx_dfc_pivot,result.dispersion_derivative,result.k_betta, amplitude_spectrum);
		growthrate.external_memory_init(vdf_diffusor.f_curr_full,vdf_diffusor.x_dfc,vdf_diffusor.y_dfc,vdf_diffusor.xy_dfc,vdf_diffusor.yx_dfc);
		//growthrate.growth_rate_update();

		
		for (unsigned count = 0; count != 10000; ++count) {
			vdf_diffusor.step();
			cudaDeviceSynchronize();
			//growthrate.growth_rate_update();
		}

		growthrate.growth_rate_update();

		if (cudaSuccess != (cudaStatus = cudaMemcpy(vdf_vector.data(), vdf_diffusor.f_curr_full, collapsed_size(&vdf_table.bounds) * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

		{
			ofstream ascii_os("./data/vdf_result.txt");
			ascii_os.exceptions(ios::failbit | ios::badbit);
			ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
			ascii_os << vdf_table;
		}

		/*{
			UniformSimpleTable<float, 2u, 1u> x_dfc_table;
			vector<float> x_dfc_vector;
			{
				x_dfc_table.bounds = vdf_table.bounds;
				x_dfc_table.space = vdf_table.space;
				x_dfc_vector.resize(collapsed_size(&x_dfc_table.bounds));
				x_dfc_table.data = x_dfc_vector.data();
			}

			if (cudaSuccess != (cudaStatus = cudaMemcpy(x_dfc_vector.data(), vdf_diffusor.xy_dfc, collapsed_size(&x_dfc_table.bounds) * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

			{
				ofstream ascii_os("./data/x_dfc_result.txt");
				ascii_os.exceptions(ios::failbit | ios::badbit);
				ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
				ascii_os << x_dfc_table;
			}
		}

		{
			UniformSimpleTable<float, 2u, 1u> y_dfc_table;
			vector<float> y_dfc_vector;
			{
				y_dfc_table.bounds = vdf_table.bounds;
				y_dfc_table.space.axes[0] = vdf_table.space.axes[1];
				y_dfc_table.space.axes[1] = vdf_table.space.axes[0];
				y_dfc_vector.resize(collapsed_size(&y_dfc_table.bounds));
				y_dfc_table.data = y_dfc_vector.data();
			}

			if (cudaSuccess != (cudaStatus = cudaMemcpy(y_dfc_vector.data(), vdf_diffusor.yx_dfc, collapsed_size(&y_dfc_table.bounds) * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

			{
				ofstream ascii_os("./data/y_dfc_result.txt");
				ascii_os.exceptions(ios::failbit | ios::badbit);
				ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
				ascii_os << y_dfc_table;
			}
		}*/

		{
			UniformSimpleTable<float, 1u, 1u> gr_table;
			vector<float> gr_vector;
			{
				gr_table.bounds.components[0] = vdf_table.bounds.components[0];
				gr_table.space.axes[0] = vdf_table.space.axes[0];
				gr_vector.resize(collapsed_size(&gr_table.bounds));
				gr_table.data = gr_vector.data();
			}

			if (cudaSuccess != (cudaStatus = cudaMemcpy(gr_vector.data(), growthrate.growth_rate_spectrum, collapsed_size(&gr_table.bounds) * sizeof(float), cudaMemcpyDeviceToHost))) throw DeviceException(cudaStatus);

			{
				ofstream ascii_os("./data/gr_result.txt");
				ascii_os.exceptions(ios::failbit | ios::badbit);
				ascii_os.precision(7); ascii_os.setf(ios::scientific, ios::floatfield);
				ascii_os << gr_table;
			}
		}
	}
	catch (exception const &ex) {
		cout << ex.what() << endl;
		return -1;
	}

	return 0;
}