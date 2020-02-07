#pragma once
#ifndef PhysicalParameters_H
#define PhysicalParameters_H

namespace iki { namespace whfi { 
	template <typename T>
	struct PhysicalParameters {
		//fundamental parameters
		T nc;               //core particles density
		T TcTh_ratio;       //ratio of the core temperature to the halo temperature
		T betta_c;          //ratio of the core thermal pressure to the magnetic pressure
		T bulk_to_alfven_c; //bulk speed in terms of alfven speed

		//derived parameters
		T nh;
		T TcTh_ratio_root, betta_root_c, betta_root_h;     //square roots of the temperature ratio and betta parameters
		T bulk_to_term_c, bulk_to_term_h;                  //bulk velocity in terms of thermal speed
	};

	template <typename T>
	PhysicalParameters<T> init_parameters(T nc, T betta_c, T TcTh_ratio, T bulk_to_alfven_c) {
		PhysicalParameters<T> p;
		p.nc = nc;
		p.betta_c = betta_c;
		p.TcTh_ratio = TcTh_ratio;
		p.bulk_to_alfven_c = bulk_to_alfven_c;

		p.nh = T(1) - nc;
		p.TcTh_ratio_root = std::sqrt(TcTh_ratio);
		p.betta_root_c = std::sqrt(T(0.5) * betta_c);
		p.betta_root_h = std::sqrt(T(0.5) * betta_c) / p.TcTh_ratio_root;
		p.bulk_to_term_c = bulk_to_alfven_c / p.betta_root_c * std::sqrt(T(1. / 1836.));
		p.bulk_to_term_h = -(nc / p.nh) * p.bulk_to_term_c * TcTh_ratio_root;

		return p;
	}
} /* whfi */ } /* iki */

#endif // !PhysicalParameters_H
