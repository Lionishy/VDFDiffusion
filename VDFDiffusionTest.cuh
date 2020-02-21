#include "SimpleTwoDimensionalSolver.cuh"
#include "SimpleTable.h"

#include "Pow.h"

int main() {
	using namespace std;
	using namespace iki;
	using namespace iki::diffusion;

	size_t vparall_size = 512, vperp_size = 512;
	UniformSpace<float, 2u> v_space;
	v_space.axes[0].begin = -15.0f; //vparall
	v_space.axes[0].step = 1.3e-2f;
	v_space.axes[1].begin = 0.f;    //vperp
	v_space.axes[1].step = 5e-2f;

	float dt = 1.f; //dt = 1./omega_c
	float rparall = dt / math::pow<2u>(v_space.axes[0].step), rperp = dt / math::pow<2u>(v_space.axes[1].step);




	SimpleTwoDimensionalSolver<float> diffusion_solver(vparall_size, vperp_size, rparall, rperp, );

	return 0;
}