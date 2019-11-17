#pragma once
#ifndef ZFuncCalculation_H
#define ZFuncCalculation_H

#include "SimpleTable.h"

#include <vector>

namespace iki {
	template <typename T>
	void zfunc_calculation(UniformSimpleTable<T, 1u, 1u> &table, std::vector<T> &data, T dksi, size_t size, size_t loop_size = 1u) {
		data.resize(size);
		zfunc_tabulator(data.begin(), dksi, size, loop_size);
		table.data = data.data();
	}
} /* iki */


#endif /* ZFuncCalculation_H */