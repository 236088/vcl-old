#include "nn.h"

void Layer::init(LayerParams& lp, RenderingParams& p, RasterizeParams& rp, float* in, int row, int column) {
	lp.column = column;
	lp.row = row;
	lp.in = in;
	lp.rast = rp.out;
	cudaMalloc(&lp.W, row *  (column + 1) *sizeof(float));
	cudaMalloc(&lp.out, p.width * p.height * row * sizeof(float));
}

__global__ void LayerForwardKernel(const LayerParams lp, const RenderingParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth)return;
	int pidx = px + p.width * (py + p.height * pz);
	if (lp.rast[pidx * 4 + 3] < 1.f) return;

	for (int r = 0; r < lp.row; r++) {
		for (int c = 0; c < lp.column; c++)
		{
			lp.out[pidx * lp.row + r] += lp.W[c * lp.row + r] * lp.in[pidx * lp.column + c];
		}
		lp.out[pidx * lp.row + r] += lp.W[lp.row * lp.column + r];
		lp.out[pidx * lp.row + r] = lp.out[pidx * lp.row + r] >0.f ? lp.out[pidx * lp.row + r] :0.f;
	}
}

void Layer::forward(LayerParams& lp, RenderingParams& p) {
	cudaMemset(lp.out, 0, p.width * p.height * lp.row * sizeof(float));
	void* args[] = { &lp,&p };
	cudaLaunchKernel(LayerForwardKernel, p.grid, p.block, args, 0, NULL);
}

void Layer::init(LayerParams& lp, RenderingParams& p, float* dLdout) {
	lp.dLdout = dLdout;
	cudaMalloc(&lp.gradW, lp.row * (lp.column + 1) * sizeof(float));
	cudaMalloc(&lp.gradIn, p.width * p.height * lp.column * sizeof(float));
}

__global__ void LayerBackwardKernel(const LayerParams lp, const RenderingParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth)return;
	int pidx = px + p.width * (py + p.height * pz);
	if (lp.rast[pidx * 4 + 3] < 1.f) return;

	for (int r = 0; r < lp.row; r++) {
		float dLdout = lp.dLdout[pidx * lp.row + r] * (lp.out[pidx * lp.row + r] >0.f ? 1.f :0.f);
		for (int c = 0; c < lp.column; c++)
		{
			lp.gradIn[pidx * lp.column + c] +=dLdout  * lp.W[c * lp.row + r];
			lp.gradW[c * lp.row + r] = dLdout * lp.in[pidx * lp.column + c];
		}
		lp.gradW[lp.row * lp.column + r] = dLdout;
	}
}

void Layer::backward(LayerParams& lp, RenderingParams& p) {
	cudaMemset(lp.gradIn, 0, p.width * p.height * lp.column * sizeof(float));
	void* args[] = { &lp,&p };
	cudaLaunchKernel(LayerBackwardKernel, p.grid, p.block, args, 0, NULL);
}