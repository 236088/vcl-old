#include "filter.h"

void Filter::init(FilterParams& fp, RasterizeParams& rp, float* in, int channel, int count) {
	fp.kernel.width = rp.kernel.width;
	fp.kernel.height = rp.kernel.height;
	fp.kernel.depth = rp.kernel.depth;
	fp.kernel.channel = channel;
	fp.kernel.count = count;
	fp.kernel.in = in;
	float filter[9] = { 0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f };
	CUDA_ERROR_CHECK(cudaMalloc(&fp.kernel.filter, 9 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(fp.kernel.filter, filter, 9 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMalloc(&fp.kernel.out, rp.kernel.width * rp.kernel.height * channel * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&fp.kernel.buf, rp.kernel.width * rp.kernel.height * channel * sizeof(float)));
	fp.block = rp.block;
	fp.grid = rp.grid;
}

__global__ void FilterForwardKernel(const FilterKernelParams fp) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= fp.width || py >= fp.height || pz >= fp.depth)return;
	int pidx = px + fp.width * (py + fp.height * pz);

	int si = 0 < px ? -1 : 0;
	int ei = px < fp.width - 1 ? 1 : 0;
	int sj = 0 < py ? -1 : 0;
	int ej = py < fp.height - 1 ? 1 : 0;
	for (int i = si; i <= ei; i++) {
		for (int j = sj; j <= ej; j++) {
			for (int k = 0; k < fp.channel; k++) {
				int idx = (px + i) + fp.width * (py + j);
				fp.out[pidx * fp.channel + k] += fp.buf[idx * fp.channel + k] * fp.filter[i + 3 * j + 4];
			}
		}
	}
}

void Filter::forward(FilterParams& fp) {
	CUDA_ERROR_CHECK(cudaMemcpy(fp.kernel.buf, fp.kernel.in, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float), cudaMemcpyDeviceToDevice));
	void* args[] = { &fp.kernel };
	for (int i = 0; i < fp.kernel.count; i++) {
		CUDA_ERROR_CHECK(cudaMemset(fp.kernel.out, 0, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float)));
		CUDA_ERROR_CHECK(cudaLaunchKernel(FilterForwardKernel, fp.grid, fp.block, args, 0, NULL));
		CUDA_ERROR_CHECK(cudaMemcpy(fp.kernel.buf, fp.kernel.out, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

void Filter::init(FilterParams& fp, float* dLdout) {
	fp.grad.out= dLdout;
	CUDA_ERROR_CHECK(cudaMalloc(&fp.grad.in, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float)));
}

__global__ void FilterBackwardKernel(const FilterKernelParams fp,const  FilterKernelGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= fp.width || py >= fp.height || pz >= fp.depth)return;
	int pidx = px + fp.width * (py + fp.height * pz);

	int si = 0 < px ? -1 : 0;
	int ei = px < fp.width - 1 ? 1 : 0;
	int sj = 0 < py ? -1 : 0;
	int ej = py < fp.height - 1 ? 1 : 0;
	for (int i = si; i <= ei; i++) {
		for (int j = sj; j <= ej; j++) {
			for (int k = 0; k < fp.channel; k++) {
				int idx = (px + i) + fp.width * (py + j);
				grad.in[pidx * fp.channel + k] += fp.buf[idx * fp.channel + k] * fp.filter[i + 3 * j + 4];
			}
		}
	}
}

void Filter::backward(FilterParams& fp) {
	CUDA_ERROR_CHECK(cudaMemcpy(fp.kernel.buf, fp.grad.out, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float), cudaMemcpyDeviceToDevice));
	void* args[] = { &fp.kernel, &fp.grad };
	for (int i = 0; i < fp.kernel.count; i++) {
		CUDA_ERROR_CHECK(cudaMemset(fp.grad.in, 0, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float)));
		CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernel, fp.grid, fp.block, args, 0, NULL));
		CUDA_ERROR_CHECK(cudaMemcpy(fp.kernel.buf, fp.grad.in, fp.kernel.width * fp.kernel.height * fp.kernel.channel * sizeof(float), cudaMemcpyDeviceToDevice));
	}
}
