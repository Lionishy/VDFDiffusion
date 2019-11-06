#pragma once
#ifndef DiffusionCoefficientImport_H
#define DiffusionCoefficientImport_H

#include "SimpleTable.h"

#include <iostream>
#include <vector>

namespace iki { namespace diffusion { 
	template <typename T>
	std::istream &import_diffusion_coefficient_pivot(std::istream &binary_is, UniformSimpleTable<T, 1u, 1u> &diffusion_pivot_table, std::vector<T> &diffusion_pivot_data) {

	}
} /* diffusion */ } /* iki */

#endif /* DiffusionCoefficientImport_H */