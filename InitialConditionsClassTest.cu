
#include "PhysicalParameters.h"
#include "ZFunc.h"
#include "ZFuncImport.h"
#include "SimpleTable.h"
#include "SimpleTableIO.h"
#include "InitialConditionsClass.cuh"


#include <iostream>
#include <fstream>



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

	auto result = initial_cond_calc.recalculate(params,-15.f,1.2e-2,1024);
	cout << result.omega[0] << endl;

	return 0;
}