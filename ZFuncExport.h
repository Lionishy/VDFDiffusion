#pragma once
#ifndef ZFuncExport_H
#define ZFuncExport_H

#include "SimpleTable.h"
#include "SimpleTableIO.h"
#include "ZFuncCalculation.h"

#include <iostream>
#include <vector>

namespace iki { 
	template <typename T>
	std::ostream &ZFuncExportAscii(std::ostream &ascii_os, T dksi, size_t size, size_t loop_count = 1u) {
		using namespace std;

		UniformSimpleTable<float, 1u, 1u> zfunc_table;
		{
			zfunc_table.bounds.components[0] = size;
			zfunc_table.space.axes[0].begin = 0.f;
			zfunc_table.space.axes[0].step = dksi;
		}
		vector<float> zfunc_data;
		zfunc_calculation(zfunc_table, zfunc_data, dksi / loop_count, size, loop_count);
		ascii_os << zfunc_table;
	}

	template <typename T>
	std::ostream &ZFuncExportBinary(std::ostream &binary_os, T dksi, size_t size, size_t loop_count = 1u) {
		using namespace std;

		UniformSimpleTable<float, 1u, 1u> zfunc_table;
		{
			zfunc_table.bounds.components[0] = size;
			zfunc_table.space.axes[0].begin = 0.f;
			zfunc_table.space.axes[0].step = dksi;
		}
		vector<float> zfunc_data;
		zfunc_calculation(zfunc_table, zfunc_data, dksi / loop_count, size, loop_count);
		{
			write_binary(binary_os,zfunc_table.space);
			write_binary(binary_os,zfunc_table.bounds);
			write_binary(binary_os, zfunc_table);
		}
	}
} /* iki */

#endif /* ZFuncExport_H */