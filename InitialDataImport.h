#pragma once
#ifndef InitialDataImport_H
#define InitialDataImport_H

#include "SimpleTable.h"
#include "SimpleTableIO.h"

#include <iostream>
#include <vector>

namespace iki { namespace diffusion { 
	template <typename T>
	std::istream& import_initial_data(std::istream &binary_is, UniformSimpleTable<T, 2u, 5u> &f_dfc_table, std::vector<T> &f_dfc_data) {
		read_binary(binary_is, f_dfc_table.space);
		read_binary(binary_is, f_dfc_table.bounds);

		f_dfc_data.resize(collapsed_size<2u>(&f_dfc_table.bounds));
		f_dfc_table.data = f_dfc_data.data();
		read_binary(binary_is, f_dfc_table);

		return binary_is;
	}
} /* diffusion */ } /* iki */

#endif /* InitialDataImport_H */