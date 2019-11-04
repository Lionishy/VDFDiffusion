#pragma once
#ifndef DeviceMemory_H
#define DeviceMemory_H

#include "DeviceException.h"

#include <cuda_runtime.h>

namespace iki {
	struct DeviceMemory final {
		DeviceMemory(size_t size) : size(size), device_ptr(NULL) {	
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMalloc(&device_ptr, size)))
				throw DeviceException(cudaStatus);
		}
		~DeviceMemory() noexcept { if (NULL != device_ptr) cudaFree(device_ptr); }
		void *get() const { return device_ptr; }

		DeviceMemory(DeviceMemory const &src) = delete;
		DeviceMemory& operator=(DeviceMemory const &src) = delete;
		DeviceMemory(DeviceMemory &&src) = delete;
		DeviceMemory& operator=(DeviceMemory &&src) = delete;

		size_t const size;
		void *device_ptr;
	};
} /* iki */

#endif

