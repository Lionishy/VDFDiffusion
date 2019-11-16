#pragma once
#ifndef KahanTabulator_H
#define KahanTabulator_H

namespace iki { namespace math {
	template <typename T, typename Seq, typename Iterator>
	void kahan_tabulator_sequence(Seq seq, Iterator begin, size_t seq_size, size_t loop_size = 1u) {
		T s = T(0), c = T(0);
		for (size_t seq_counter = 0u; seq_counter != seq_size; ++seq_counter) {
			for (size_t loop_count = 0u; loop_count != loop_size; ++loop_count) {
				T y, t;
				y = seq(seq_counter * loop_size + loop_count, s) - c;
				t = s + y;
				c = (t - s) - y;
				s = t;
			}
			*begin++ = s;
		}
	}
} /* math */ } /* iki */

#endif /* KahanTabulator_H */