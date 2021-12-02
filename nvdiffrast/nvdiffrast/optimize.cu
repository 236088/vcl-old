#include "optimize.h"

void Loss::init(LossParams& loss, float* predict, float* target, int width,int height, int depth) {
	loss.predict = predict;
	loss.target = target;
	CUDA_ERROR_CHECK(cudaMalloc(&loss.grad, width * height * depth * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&loss.buffer, width * height * depth * sizeof(float)));
	loss.size = width * height * depth;
	loss.width = width;
	loss.height = height;
	loss.depth = depth;

	loss.block = getBlock(width, height);
	loss.grid = getGrid(loss.block, width, height, depth);

	int msb = 0;
	if (loss.size & 0xffff0000)msb += 16;
	if (loss.size & 0xff00ff00)msb += 8;
	if (loss.size & 0xf0f0f0f0)msb += 4;
	if (loss.size & 0xcccccccc)msb += 2;
	if (loss.size & 0xaaaaaaaa)msb += 1;
	int hmsb = msb / 2;
	--msb;
	loss.stride = 1 << msb;
	loss.lh = 1 << hmsb;
	loss.rh= 1 << (msb - hmsb);
}

__global__ void square(const LossParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + loss.width * (py + loss.height * pz);
	if (px >= loss.width || py >= loss.height || pz >= loss.depth)return;
	float diff = loss.predict[pidx] - loss.target[pidx];
	loss.buffer[pidx] = diff * diff;
}

__global__ void reduction(const LossParams loss, int width, int height, int stride) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + width * (py + height * pz);
	if (px >= loss.width || py >= loss.height || pz >= loss.depth)return;
	loss.buffer[pidx] += loss.buffer[pidx + stride];
}

float Loss::MSE(LossParams& loss) {
	void* sargs[] = { &loss };
	CUDA_ERROR_CHECK(cudaLaunchKernel(square, loss.grid, loss.block, sargs, 0, NULL));
	int stride = loss.stride, w = loss.lh, h = loss.rh;
	void* args[] = { &loss, &w, &h, &stride };
	while (stride > 0)
	{
		dim3 block = getBlock(w, h);
		dim3 grid = getGrid(block, w, h);
		CUDA_ERROR_CHECK(cudaLaunchKernel(reduction, grid, block, args, 0, NULL));
		stride >>= 1;
		if (h >= w)h >>= 1;
		else w >>= 1;
	}
	float sum;
	CUDA_ERROR_CHECK(cudaMemcpy(&sum, loss.buffer, sizeof(float), cudaMemcpyDeviceToHost));
	return sum / 2.;
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
	CUDA_ERROR_CHECK(cudaLaunchKernel(lossGradient, loss.grid, loss.block, args, 0, NULL));
}



void Optimizer::init(OptimizeParams& opt, float* param, float* grad, int size, int width, int height, int depth) {
	opt.param = param;
	opt.grad = grad;
	opt.size = size;
	opt.it = 1;
	opt.width = width;
	opt.height = height;
	opt.depth = depth;
	opt.block = getBlock(width, height);
	opt.grid = getGrid(opt.block, width, height, depth);
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

__global__ void random(const OptimizeParams opt, long seed, float min, float max) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;
	unsigned int rnd = pidx;
	opt.param[pidx] = min + (max - min) * getUniform(rnd, (unsigned int)seed, 0xdeadbeef);
}

void Optimizer::randomParams(OptimizeParams& opt, float min, float max) {
	struct timespec t;
	timespec_get(&t, TIME_UTC);
	long nsec = t.tv_nsec;
	void* args[] = { &opt ,&nsec, &min, &max };
	CUDA_ERROR_CHECK(cudaLaunchKernel(random, opt.grid, opt.block, args, 0, NULL));
}

void Adam::init(AdamParams& adam,double eta, double rhom, double rhov,  double eps) {
	adam.rhom = rhom;
	adam.rhov = rhov;
	adam.rhomt = rhom;
	adam.rhovt = rhov;
	adam.eta = eta;
	adam.eps = eps;
	CUDA_ERROR_CHECK(cudaMalloc(&adam.m, adam.size * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&adam.v, adam.size * sizeof(float)));
}

void Adam::init(AdamParams& adam, float* param, float* grad, int size, int width, int height, int depth, double eta, double rhom, double rhov, double eps) {
	Optimizer::init(adam, param, grad, size, width, height, depth);
	init(adam, eta, rhom, rhov, eps);
}

void Adam::init(AdamParams& adam, Attribute& attr, float* grad, double eta, double rhom, double rhov, double eps) {
	int size = attr.vboNum * attr.dimention;
	Optimizer::init(adam, attr.vbo, grad, size, attr.vboNum, attr.dimention, 1);
	init(adam, eta,rhom, rhov,  eps);
}

__global__ void adamStep(const AdamParams adam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + adam.width * (py + adam.height * pz);
	if (pidx >= adam.size)return;

	adam.m[pidx] = adam.rhom * adam.m[pidx] + (1 - adam.rhom) * adam.grad[pidx];
	adam.v[pidx] = adam.rhov * adam.v[pidx] + (1 - adam.rhov) * adam.grad[pidx] * adam.grad[pidx];
	double m = adam.m[pidx] / (1 - adam.rhomt);
	double v = adam.v[pidx] / (1 - adam.rhovt);
	AddNaNcheck(adam.param[pidx], -m * adam.eta / (sqrt(v) + adam.eps));
}

void Adam::step(AdamParams& adam) {
	adam.it++;
	adam.rhomt *= adam.rhom;
	adam.rhovt *= adam.rhov;
	void* args[] = { &adam };
	CUDA_ERROR_CHECK(cudaLaunchKernel(adamStep, adam.grid, adam.block, args, 0, NULL));
}

void Nadam::init(NadamParams& nadam, double alpha,  double mu, double rho, double eps) {
	nadam.mupow = pow(0.96, 0.004);
	nadam.mupowt = nadam.mupow;
	nadam.mu = mu;
	nadam.mu0 = mu * (1 - .5 * nadam.mupowt);
	nadam.mupowt *= nadam.mupow;
	nadam.mu1 = mu * (1 - .5 * nadam.mupowt);
	nadam.rho = rho;
	nadam.alpha = alpha;
	nadam.mut0 = nadam.mu0;
	nadam.mut1 = nadam.mu0 * nadam.mu1;
	nadam.rhot = rho;
	nadam.eps = eps;
	CUDA_ERROR_CHECK(cudaMalloc(&nadam.m, nadam.size * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&nadam.v, nadam.size * sizeof(float)));
}

void Nadam::init(NadamParams& nadam, float* param, float* grad, int size, int width, int height, int depth,  double alpha, double mu,double rho, double eps) {
	Optimizer::init(nadam, param, grad, size, width, height, depth);
	init(nadam,alpha,  mu, rho, eps);
}

void Nadam::init(NadamParams& nadam, Attribute& attr, float* grad, double alpha, double mu, double rho, double eps) {
	int size = attr.vboNum * attr.dimention;
	Optimizer::init(nadam, attr.vbo, grad, size, attr.vboNum, attr.dimention, 1);
	init(nadam, alpha, mu, rho, eps);
}

__global__ void nadamStep(const NadamParams nadam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + nadam.width * (py + nadam.height * pz);
	if (pidx >= nadam.size)return;

	nadam.m[pidx] = nadam.mu * nadam.m[pidx] + (1 - nadam.mu) * nadam.grad[pidx];
	nadam.v[pidx] = nadam.rho * nadam.v[pidx] + (1 - nadam.rho) * nadam.grad[pidx] * nadam.grad[pidx];
	double m =  nadam.mu1 / (1 - nadam.mut1) *nadam.m[pidx] + (1 - nadam.mu0) / (1- nadam.mut0) * nadam.grad[pidx];
	double v = 1 / (1 - nadam.rhot) * nadam.v[pidx];
	AddNaNcheck(nadam.param[pidx], -m * nadam.alpha / (sqrt(v) + nadam.eps));
}

void Nadam::step(NadamParams& nadam) {
	nadam.it++;
	nadam.mu0 = nadam.mu1;
	nadam.mut0 = nadam.mut1;
	nadam.mupowt *= nadam.mupow;
	nadam.mu1 = nadam.mu * (1 - .5 * nadam.mupowt);
	nadam.mut1 *= nadam.mu1;
	nadam.rhot *= nadam.rho;
	void* args[] = { &nadam };
	CUDA_ERROR_CHECK(cudaLaunchKernel(nadamStep, nadam.grid, nadam.block, args, 0, NULL));
}