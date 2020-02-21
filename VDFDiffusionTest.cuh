#include "SimpleTwoDimensionalSolver.cuh"
#include "SimpleTable.h"

#include "Pow.h"

#include <vector>

int main() {
	using namespace std;
	using namespace iki;
	using namespace iki::diffusion;

	UniformSimpleTable<float, 2u, 1u> vdf_table;
	vector<float> vdf_table_data;
	{ //initialize vdf_table
		vdf_table.bounds.components[0] = vdf_table.bounds.components[1] = 512;
		vdf_table.space.axes[0].begin = -15.0f; //vparall
		vdf_table.space.axes[0].step = 1.3e-2f;
		vdf_table.space.axes[1].begin = 0.f;    //vperp
		vdf_table.space.axes[1].step = 5e-2f;
		vdf_table_data.resize(collapsed_size(&vdf_table.bounds));
		vdf_table.data = vdf_table_data.data();
	}

	float dt = 1.f; //dt = 1./omega_c
	float rparall = dt / math::pow<2u>(vdf_table.space.axes[0].step), rperp = dt / math::pow<2u>(vdf_table.space.axes[1].step);




	SimpleTwoDimensionalSolver<float> vdf_diffusor(vdf_table.bounds.components[0], vdf_table.bounds.components[1], rparall, rperp, );

	return 0;
}