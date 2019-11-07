#pragma once
#ifndef DiffusionCoefficientImport_H
#define DiffusionCoefficientImport_H

#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <iostream>
#include <vector>

namespace iki { namespace diffusion { 
	template <typename T>
	std::istream &import_diffusion_coefficient_pivot(std::istream &binary_is, UniformSimpleTable<T, 1u, 1u> &diffusion_pivot_table, std::vector<T> &diffusion_pivot_data) {
		read_binary(binary_is, diffusion_pivot_table.space);
		read_binary(binary_is, diffusion_pivot_table.bounds);

		diffusion_pivot_data.resize(diffusion_pivot_table.bounds.components[0]);
		diffusion_pivot_table.data = zfunc_data.data();
		read_binary(binary_is, diffusion_pivot_table);

		return binary_is;
	}
} /* diffusion */ } /* iki */

#endif /* DiffusionCoefficientImport_H */