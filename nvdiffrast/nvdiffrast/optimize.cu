#include "optimize.h"

void Loss::init(LossParams& p, RenderBufferGrad& predict, RenderBuffer& target) {
	p.params.width = predict.width;
	p.params.height = predict.height;
	p.params.channel = predict.channel;
	p.params.depth = predict.depth;
	p.params.predict = predict.buffer;
	p.params.target = target.buffer;
	p.params.grad = predict.grad;

	p.block = getBlock(predict.width, predict.height, predict.channel);
	p.grid = getGrid(p.block, predict.width, predict.height, predict.depth);
}


void MSELoss::init(MSELossParams& p, RenderBufferGrad& predict, RenderBuffer& target) {
	Loss::init(p, predict, target);
	cudaMalloc(&p.mse.buf, predict.Size());
	p.mse.length = predict.Length();
	size_t stride = 0;
	stride = stride | (stride >> 1);
	stride = stride | (stride >> 2);
	stride = stride | (stride >> 4);
	stride = stride | (stride >> 8);
	stride = stride | (stride >> 16);
	stride = stride | (stride >> 32);
	stride ^= (stride >> 1);
	int msb = 0;
	if (msb & 0xffffffff00000000)msb += 32;
	if (msb & 0xffff0000ffff0000)msb += 16;
	if (msb & 0xff00ff00ff00ff00)msb += 8;
	if (msb & 0xf0f0f0f0f0f0f0f0)msb += 4;
	if (msb & 0xcccccccccccccccc)msb += 2;
	if (msb & 0xaaaaaaaaaaaaaaaa)msb += 1;
	p.stride = stride;
	int hmsb = msb / 2;
	p.lh = stride >> hmsb;
	p.rh = 1 << hmsb;
}

__global__ void square(const LossKernelParams p, const MSELossKernelParams mse) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pc = threadIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth || pc >= p.channel)return;
	int pidx = pc + p.channel * (px + p.width * (py + p.height * pz));
	mse.buf[pidx] = p.predict[pidx] - p.target[pidx];
	mse.buf[pidx] *= mse.buf[pidx];
}

__global__ void reduction(const MSELossKernelParams mse, size_t stride, int rh) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	size_t pidx = px + (size_t)rh * py;
	if (pidx + stride < mse.length) {
		mse.buf[pidx] += mse.buf[pidx + stride];
	}
}

float MSELoss::loss(MSELossParams& p) {
	void* sargs[] = { &p.params, &p.mse };
	size_t stride = p.stride;
	int lh = p.lh, rh = p.rh;
	void* args[] = { &p.mse, &stride,&rh };
	while (stride>0){
		dim3 block = getBlock(lh, rh);
		dim3 grid = getGrid(block, lh, rh, 1);
		cudaLaunchKernel(reduction, grid, block, args, 0, NULL);
		stride >>= 1;
		if (lh > rh)lh >>= 1;
		else rh >>= 1;
	}
	float sum;
	cudaMemcpy(&sum, p.mse.buf, sizeof(float), cudaMemcpyDeviceToHost);
	return sum/float(p.mse.length);
}

__global__ void lossGradient(const LossKernelParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pc = threadIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth || pc >= p.channel)return;
	int pidx = pc + p.channel * (px + p.width * (py + p.height * pz));
	p.grad[pidx] = p.predict[pidx] - p.target[pidx];
}

void MSELoss::backward(MSELossParams& p) {
	void* args[] = { &p.params };
	cudaLaunchKernel(lossGradient, p.grid, p.block, args, 0, NULL);
}



void Optimize::init(OptimizeParams& p, float* param, float* grad, int size, int width, int height, int depth, int channel) {
	p.params.param = param;
	p.params.grad = grad;
	p.it = 1;
	p.params.size = size;
	p.params.width = width;
	p.params.height = height;
	p.params.depth = depth;
	p.params.channel = channel;
	p.block = getBlock(width, height, channel);
	p.grid = getGrid(p.block, width, height, depth);
}

void Optimize::init(OptimizeParams& p, AttributeGrad& attr) {
	init(p, attr.vbo, attr.grad, attr.vboLength(), attr.vboNum, 1, 1, attr.dimention);
}

void Optimize::init(OptimizeParams& p, MipTextureGrad& texture) {
	init(p, texture.texture[0], texture.grad[0], texture.Length(0), texture.width, texture.height, texture.channel, 1);
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

__global__ void random( const OptimizeKernelParams p, long seed, float min, float max) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pc = threadIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth || pc >= p.channel)return;
	int pidx = pc + p.channel * (px + p.width * (py + p.height * pz));
	unsigned int rnd = pidx;
	p.param[pidx] += min + (max - min) * getUniform(rnd, (unsigned int)seed, 0xdeadbeef);
}

void Optimize::addRandomParams(OptimizeParams& p, float min, float max) {
	struct timespec t{};
	long nsec = t.tv_nsec;
	void* args[] = { &p.params ,&nsec, &min, &max};
	cudaLaunchKernel(random , p.grid, p.block, args, 0, NULL);
}

void Adam::init(AdamParams& p, float* param, float* grad, int size, int width, int height, int depth, int channel, double rhom, double rhov, double eta, double eps) {
	Optimize::init((OptimizeParams&)p, param, grad, size, width, height, depth, channel);
	p.adam.rhom = rhom;
	p.adam.rhov = rhov;
	p.adam.rhomt = rhom;
	p.adam.rhovt = rhov;
	p.adam.eta = eta;
	p.adam.eps = eps;
	cudaMalloc(&p.adam.m, size * sizeof(float));
	cudaMalloc(&p.adam.v, size * sizeof(float));
}

void Adam::init(AdamParams& p, AttributeGrad& attr, double rhom, double rhov, double eta, double eps) {
	init(p, attr.vbo, attr.grad, attr.vboLength(), attr.vboNum, 1, 1, attr.dimention, rhom, rhov, eta, eps);
}

void Adam::init(AdamParams& p, MipTextureGrad& texture, double rhom, double rhov, double eta, double eps) {
	init(p, texture.texture[0], texture.grad[0], texture.Length(0), texture.width, texture.height, texture.channel, 1, rhom, rhov, eta, eps);
}

__global__ void adamStep(const OptimizeKernelParams p, const AdamKernelParams adam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pc = threadIdx.z;
	if (px >= p.width || py >= p.height || pz >= p.depth || pc >= p.channel)return;
	int pidx = pc + p.channel * (px + p.width * (py + p.height * pz));

	adam.m[pidx] = adam.rhom * adam.m[pidx] + (1.0 - adam.rhom) * p.grad[pidx];
	adam.v[pidx] = adam.rhov * adam.v[pidx] + (1.0 - adam.rhov) * p.grad[pidx] * p.grad[pidx];
	double m = adam.m[pidx] / (1.0 - adam.rhomt);
	double v = adam.v[pidx] / (1.0 - adam.rhovt);
	AddNaNcheck(p.param[pidx], -adam.eta / sqrt(v + adam.eps) * m);
}

void Adam::step(AdamParams& p) {
	p.it++;
	p.adam.rhomt *= p.adam.rhom;
	p.adam.rhovt *= p.adam.rhov;
	void* args[] = { &p.params ,&p.adam};
	cudaLaunchKernel(adamStep, p.grid, p.block, args, 0, NULL);
}