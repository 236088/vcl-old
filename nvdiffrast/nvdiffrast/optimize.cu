#include "optimize.h"

__global__ void square(const LossParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= loss.width || py >= loss.height || pz >= loss.depth)return;
	int pidx = px + loss.width * (py + loss.height * pz);
	loss.buffer[pidx] = loss.predict[pidx] - loss.target[pidx];
	loss.buffer[pidx] *= loss.buffer[pidx];
}

__global__ void reduction(const LossParams loss, int width, int height, int stride) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + width * (py + height * pz);
	loss.buffer[pidx] += loss.buffer[pidx + stride];
}

int msb(int v)
{
	int b = 0;
	if (v & 0xffff0000)b += 16;
	if (v & 0xff00ff00)b += 8;
	if (v & 0xf0f0f0f0)b += 4;
	if (v & 0xcccccccc)b += 2;
	if (v & 0xaaaaaaaa)b += 1;
	return b;
}

float Loss::MSE(LossParams& loss) {
	square << <loss.grid, loss.block >> > (loss);
	int b = msb(loss.size);
	int stride = 1 << (b - 1);
	int w = 1 << (b / 2), h = 1 << (b - 1 - b / 2);
	while (stride>0)
	{
		dim3 block = getBlock(w, h);
		dim3 grid = getGrid(block, w, h);
		reduction << <grid, block >> > (loss, w, h, stride);
		stride >>= 1;
		if (h>=w)h >>= 1;
		else w >>= 1;
	}
	float sum;
	cudaMemcpy(&sum, loss.buffer, sizeof(float), cudaMemcpyDeviceToHost);
	return sum / loss.size;
}

void Loss::init(LossParams& loss, float* predict, float* target, RenderingParams& p, int dimention) {
	loss.predict = predict;
	loss.target = target;
	cudaMalloc(&loss.grad, p.width * p.height * dimention * sizeof(float));
	cudaMalloc(&loss.buffer, p.width * p.height * dimention * sizeof(float));
	loss.size = p.width * p.height * dimention;
	loss.width = p.width;
	loss.height = p.height;
	loss.depth = dimention;

	int width = p.width, height = p.height;
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
	loss.grid = dim3(w, h, dimention);
}

__global__ void lossGradient(const LossParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + loss.width * (py + loss.height * pz);
	if (px >= loss.width || py >= loss.height || pz >= loss.depth)return;
	loss.grad[pidx] = loss.predict[pidx] - loss.target[pidx];
}

void Loss::backward(LossParams& loss) {
	void* args[] = { &loss };
	cudaLaunchKernel(lossGradient, loss.grid, loss.block, args, 0, NULL);
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

void Adam::init(AdamParams& adam, Attribute& attr, int width, int height, int depth, double rhom, double rhov, double eta, double eps) {
	adam.param = attr.vbo;
	adam.grad = attr.grad;
	adam.it = 1;
	adam.rhom = rhom;
	adam.rhov = rhov;
	adam.rhomt = rhom;
	adam.rhovt = rhov;
	adam.eta = eta;
	adam.eps = eps;
	int size = attr.vboNum * attr.dimention;
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

__device__ __forceinline__ float getUniform(unsigned int a, unsigned  int b, unsigned int c)
{
	a -= b; a -= c; a ^= (c >> 13);
	b -= c; b -= a; b ^= (a << 8);
	c -= a; c -= b; c ^= (b >> 13);
	a -= b; a -= c; a ^= (c >> 12);
	b -= c; b -= a; b ^= (a << 16);
	c -= a; c -= b; c ^= (b >> 5);
	a -= b; a -= c; a ^= (c >> 3);
	b -= c; b -= a; b ^= (a << 10);
	c -= a; c -= b; c ^= (b >> 15);
	int d = 0x007fffff; 
	return (float)(c & d) / (float)d;
}

__global__ void random( const AdamParams adam, long seed) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + adam.width * (py + adam.height * pz);
	if (pidx >= adam.size)return;
	unsigned int rnd = pidx;
	adam.param[pidx] = getUniform(rnd, (unsigned int)seed, 0xdeadbeef);
}

void Adam::randomParams(AdamParams& adam) {
	struct timespec t{};
	long nsec = t.tv_nsec;
	void* args[] = { &adam ,&nsec};
	cudaLaunchKernel(random , adam.grid, adam.block, args, 0, NULL);
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
	AddNaNcheck(adam.param[pidx], -adam.eta / sqrt(v + adam.eps) * m);
}

void Adam::step(AdamParams& adam) {
	adam.it++;
	adam.rhomt *= adam.rhom;
	adam.rhovt *= adam.rhov;
	void* args[] = { &adam };
	cudaLaunchKernel(adamStep, adam.grid, adam.block, args, 0, NULL);
}