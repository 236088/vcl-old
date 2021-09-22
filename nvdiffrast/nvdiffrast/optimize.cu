#include "optimize.h"


void Loss::MSE(LossParams& loss) {

}

void Loss::init(LossParams& loss, float* predict, float* correct, int size, int width, int height, int depth) {
	loss.predict = predict;
	loss.correct = correct;
	cudaMalloc(&loss.grad, size * sizeof(float));
	loss.size = size;
	loss.width = width;
	loss.height = height;
	loss.depth = depth;

	int w = 1, h = 1;
	if (width > MAX_DIM_PER_BLOCK) {
		w = (width + MAX_DIM_PER_BLOCK - 1) / MAX_DIM_PER_BLOCK;
		width = MAX_DIM_PER_BLOCK;
	}
	if (height > MAX_DIM_PER_BLOCK) {
		h = (height + MAX_DIM_PER_BLOCK - 1) / MAX_DIM_PER_BLOCK;
		height = MAX_DIM_PER_BLOCK;
	}
	loss.block = dim3(width, height);
	loss.grid = dim3(w, h, depth);
}

__global__ void lossGradient(const LossParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + loss.width * (py + loss.height * pz);
	if (pidx >= loss.size)return;
	loss.grad[pidx] = loss.predict[pidx] - loss.correct[pidx];
}

void Loss::backward(LossParams& loss) {
	lossGradient << <loss.grid, loss.block >> > (loss);
}

void Adam::init(AdamParams& adam, float* param, float* grad, int size, int width, int height, int depth, double rhom, double rhov, double eta, double eps) {
	adam.param = param;
	adam.grad = grad;
	adam.it = 1;
	adam.rhom = rhom;
	adam.rhov = rhov;
	adam.rhomt = rhom;
	adam.rhovt = rhov;
	adam.eta = eta;
	adam.eps = eps;
	cudaMalloc(&adam.m, size * sizeof(float));
	cudaMemset(adam.m, 0, size * sizeof(float));
	cudaMalloc(&adam.v, size * sizeof(float));
	cudaMemset(adam.v, 0, size * sizeof(float));
	adam.size = size;
	adam.width = width;
	adam.height = height;
	adam.depth = depth;
	int w = 1, h = 1;
	if (width > MAX_DIM_PER_BLOCK) {
		w = (width + MAX_DIM_PER_BLOCK - 1) / MAX_DIM_PER_BLOCK;
		width = MAX_DIM_PER_BLOCK;
	}
	if (height > MAX_DIM_PER_BLOCK) {
		h = (height + MAX_DIM_PER_BLOCK - 1) / MAX_DIM_PER_BLOCK;
		height = MAX_DIM_PER_BLOCK;
	}
	adam.block = dim3(width, height);
	adam.grid = dim3(w, h, depth);
}

__global__ void adamStep(const AdamParams adam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + adam.width * (py + adam.height * pz);
	if (pidx >= adam.size)return;

	adam.m[pidx] = adam.rhom * adam.m[pidx] + (1.0 - adam.rhom) * adam.grad[pidx];
	adam.v[pidx] = adam.rhov * adam.v[pidx] + (1.0 - adam.rhov) * adam.grad[pidx] * adam.grad[pidx];
	double m = adam.m[pidx] / (1.0 - adam.rhomt);
	double v = adam.v[pidx] / (1.0 - adam.rhovt);
	adam.param[pidx] -= adam.eta / sqrt(v + adam.eps) * m;
}

void Adam::step(AdamParams& adam) {
	adam.it++;
	adam.rhomt *= adam.rhom;
	adam.rhovt *= adam.rhov;
	adamStep << <adam.grid, adam.block >> > (adam);
}