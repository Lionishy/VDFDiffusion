#pragma once
#ifndef DeviceException_H
#define DeviceException_H

#include <cuda_runtime.h>
#include <stdexcept>

namespace iki {
	struct DeviceException: std::runtime_error {
		DeviceException(cudaError_t cudaStatus): std::runtime_error(cudaGetErrorString(cudaStatus)), cudaStatus(cudaStatus) { }
		cudaError_t cudaStatus;
	};
} /* iki */

#endif