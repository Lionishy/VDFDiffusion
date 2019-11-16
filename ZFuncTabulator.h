#pragma once

#include "ZFuncRungeKuttaSequence.h"
#include "KahanTabulator.h"

namespace iki {
	template <typename T, typename Iterator>
	void zfunc_tabulator(Iterator begin, T dksi, size_t size, size_t loop_size = 1u) {
		ZFuncRungeKuttaSequence seq(dksi);
		math::kahan_tabulator_sequence(seq, size, begin, loop_size);
	}
} /* iki */