
#include "PhysicalParameters.h"
#include "ZFunc.h"
#include "ZFuncImport.h"
#include "SimpleTable.h"
#include "SimpleTableIO.h"
#include "InitialConditionsClass.cuh"


#include <iostream>
#include <fstream>
#include <algorithm>


int main() {
	using namespace std;
	using namespace iki;

	whfi::PhysicalParameters<float> params = whfi::init_parameters(0.85f, 1.f / 0.85f, 0.25f, -9.f);

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

	auto result = initial_cond_calc.recalculate(params,-15.f,1.4e-2f,1024);
	UniformSimpleTable<float, 1u, 3u> diffusion_table;
	diffusion_table.space.axes[0].begin = -15.f;
	diffusion_table.space.axes[0].step = 1.3e-2f;
	diffusion_table.bounds.components[0] = 1024;
	
	vector<float> diffusion;
	for (unsigned idx = 0; idx != result.Dpl.size(); ++idx) {
		diffusion.push_back(result.Dpl[idx]);
		diffusion.push_back(result.Dmx[idx]);
		diffusion.push_back(result.Dpr[idx]);
	}

	diffusion_table.data = diffusion.data();

	{
		ofstream ascii_os("./data/diffusion.txt");
		ascii_os.precision(7); ascii_os.setf(ios::floatfield, ios::fixed);
		ascii_os << diffusion_table;
	}


	return 0;
}