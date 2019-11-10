#pragma once
#ifndef GammaPivotImport_H
#define GammaPivotImport_H

#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <iostream>
#include <vector>

namespace iki { namespace diffusion { 
	template <typename T>
	std::istream &import_gamma_pivot(std::istream &binary_is, UniformSimpleTable<T, 1u, 2u> &gamma_pivot_table, std::vector<T> &gamma_pivot_data) {
		read_binary(binary_is, gamma_pivot_table.space);
		read_binary(binary_is, gamma_pivot_table.bounds);

		gamma_pivot_data.resize(collapsed_size<1u>(&gamma_pivot_table.bounds));
		gamma_pivot_table.data = gamma_pivot_data.data();
		read_binary(binary_is, gamma_pivot_table);

		return binary_is;
	}
} /* diffusion */ } /* iki */

#endif /* GammaPivotImport_H */