#pragma once
#ifndef StepSolver_H
#define StepSolver_H

#include <exception>
#include <stdexcept>

namespace iki { namespace math { 
	template <typename T, typename Eqn_t>
	T step_solver(Eqn_t f, T start, T step, T stop) {
		unsigned count = 0; T arg_curr = start + step * count, arg_next = start + step * (count + 1);
		T f_curr = f(arg_curr)
		, f_next = f(arg_next);

		while (arg_curr < stop) {
			if (f_curr * f_next < T(0)) return T(0.5) * (arg_curr + arg_next);

			arg_curr = arg_next;
			arg_next = start + step * ++count;
			f_curr = f_next;
			f_next = f(arg_next);
		}

		throw std::runtime_error("Step solver: no root found");
	}
} /*math*/ } /*iki*/

#endif /*StepSolver_H*/