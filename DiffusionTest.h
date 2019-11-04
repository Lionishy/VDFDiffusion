#pragma once
#ifndef DiffusionTest_H
#define DiffusionTest_H

#include <vector>
#include <cmath>

namespace iki {	namespace diffusion {
	template <typename T>
	void x_slope_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, size_t x_size, size_t y_size) {
		{
			T grad = T(1) / (y_size - 1);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = T(1) - grad * x_idx;
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 1; y_idx != y_size-2; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size-2; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void y_slope_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, size_t x_size, size_t y_size) {
		{
			T grad = T(1) / (y_size - 1);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = T(1) - grad * y_idx;
		}

		{
			for (size_t y_idx = 1; y_idx != y_size-2; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size-2; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void x_sin_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, size_t x_size, size_t y_size, int Nx) {
		{
			auto const PI = T(3.14159265358979323);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = std::sin((PI * Nx) / (x_size - 1) * x_idx);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void y_sin_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, size_t x_size, size_t y_size, int Ny) {
		{
			auto const PI = T(3.14159265358979323);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = std::sin((PI * Ny) / (y_size - 1) * y_idx);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void x_y_sin_sin_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, size_t x_size, size_t y_size, int Nx, int Ny) {
		{
			auto const PI = T(3.14159265358979323);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = std::sin((PI * Nx) / (x_size - 1) * x_idx) * std::sin((PI * Ny) / (y_size - 1) * y_idx);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void x_slope_mixed_term_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, std::vector<T> &xy_dfc, std::vector<T> &yx_dfc, size_t x_size, size_t y_size) {
		{
			T grad = T(1) / (y_size - 1);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = T(1) - grad * x_idx;
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 1; y_idx != y_size - 2; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size - 2; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 2; y_idx != y_size - 3; ++y_idx)
				for (size_t x_idx = 2; x_idx != x_size - 3; ++x_idx)
					xy_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}

	template <typename T>
	void y_slope_mixed_term_test(std::vector<T> &f, std::vector<T> &x_dfc, std::vector<T> &y_dfc, std::vector<T> &xy_dfc, std::vector<T> &yx_dfc, size_t x_size, size_t y_size) {
		{
			T grad = T(1) / (y_size - 1);
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 0; x_idx != x_size; ++x_idx)
					f[x_idx * y_size + y_idx] = T(1) - grad * y_idx;
		}

		{
			for (size_t y_idx = 1; y_idx != y_size - 2; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size - 2; ++x_idx)
					x_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 0; y_idx != y_size; ++y_idx)
				for (size_t x_idx = 1; x_idx != x_size; ++x_idx)
					y_dfc[y_idx + x_idx * y_size] = T(1);
		}

		{
			for (size_t y_idx = 2; y_idx != y_size - 3; ++y_idx)
				for (size_t x_idx = 2; x_idx != x_size - 3; ++x_idx)
					yx_dfc[y_idx + x_idx * y_size] = T(1);
		}
	}
} /* diffusion */ } /* iki */

#endif /* DiffusionTest_H */