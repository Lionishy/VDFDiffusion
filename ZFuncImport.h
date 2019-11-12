#pragma once
#ifndef ZFuncImport_H
#define ZFuncImport_H

#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <iostream>
#include <vector>

namespace iki {
	template <typename T>
	std::istream &ZFuncImport(std::istream &binary_is, iki::UniformSimpleTable<T, 1u, 1u> &zfunc_table, std::vector<T> &zfunc_data) {
		read_binary(binary_is, zfunc_table.space);
		read_binary(binary_is, zfunc_table.bounds);

		zfunc_data.resize(zfunc_table.bounds.components[0]);
		zfunc_table.data = zfunc_data.data();
		read_binary(binary_is, zfunc_table);
		return binary_is;
	}
} /* iki */

#endif /* ZFuncImport_H */