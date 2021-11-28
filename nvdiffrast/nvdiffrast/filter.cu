#include "filter.h"

void Filter::init(FilterParams& fp, RenderingParams& p, float* in, int channel, int count) {
	fp.channel = channel;
	fp.count = count;
	fp.in = in;
	float filter[9] = { 0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f };
	cudaMalloc(&fp.filter, 9 * sizeof(float));
	cudaMemcpy(fp.filter, filter, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&fp.out, p.width * p.height * channel * sizeof(float));
	cudaMalloc(&fp.buf, p.width * p.height * channel * sizeof(float));
}

__global__ void FilterForwardKernel(const FilterParams fp, const RenderingParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth)return;
	int pidx = px + p.width * (py + p.height * pz);

	int si = 0 < px ? -1 : 0;
	int ei = px < p.width - 1 ? 1 : 0;
	int sj = 0 < py ? -1 : 0;
	int ej = py < p.height - 1 ? 1 : 0;
	for (int i = si; i <= ei; i++) {
		for (int j = sj; j <= ej; j++) {
			for (int k = 0; k < fp.channel; k++) {
				int idx = (px + i) + p.width * (py + j);
				fp.out[pidx * fp.channel + k] += fp.buf[idx * fp.channel + k] * fp.filter[i + 3 * j + 4];
			}
		}
	}
}

void Filter::forward(FilterParams& fp, RenderingParams& p) {
	cudaMemcpy(fp.buf, fp.in, p.width * p.height * fp.channel * sizeof(float), cudaMemcpyDeviceToDevice);
	void* args[] = { &fp, &p };
	for (int i = 0; i < fp.count; i++) {
		cudaMemset(fp.out, 0, p.width * p.height * fp.channel * sizeof(float));
		cudaLaunchKernel(FilterForwardKernel, p.grid, p.block, args, 0, NULL);
		cudaMemcpy(fp.buf, fp.out, p.width * p.height * fp.channel * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}

void Filter::init(FilterParams& fp, RenderingParams& p, float* dLdout) {
	fp.dLdout= dLdout;
	cudaMalloc(&fp.gradIn, p.width * p.height * fp.channel * sizeof(float));
}

__global__ void FilterBackwardKernel(const FilterParams fp, const RenderingParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth)return;
	int pidx = px + p.width * (py + p.height * pz);

	int si = 0 < px ? -1 : 0;
	int ei = px < p.width - 1 ? 1 : 0;
	int sj = 0 < py ? -1 : 0;
	int ej = py < p.height - 1 ? 1 : 0;
	for (int i = si; i <= ei; i++) {
		for (int j = sj; j <= ej; j++) {
			for (int k = 0; k < fp.channel; k++) {
				int idx = (px + i) + p.width * (py + j);
				fp.gradIn[pidx * fp.channel + k] += fp.buf[idx * fp.channel + k] * fp.filter[i + 3 * j + 4];
			}
		}
	}
}

void Filter::backward(FilterParams& fp, RenderingParams& p) {
	cudaMemcpy(fp.buf, fp.dLdout, p.width * p.height * fp.channel * sizeof(float), cudaMemcpyDeviceToDevice);
	void* args[] = { &fp, &p };
	for (int i = 0; i < fp.count; i++) {
		cudaMemset(fp.gradIn, 0, p.width * p.height * fp.channel * sizeof(float));
		cudaLaunchKernel(FilterBackwardKernel, p.grid, p.block, args, 0, NULL);
		cudaMemcpy(fp.buf, fp.gradIn, p.width * p.height * fp.channel * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}
