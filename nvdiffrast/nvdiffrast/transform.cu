#include "transform.h"

void Transform::init(TransformParams& p,  float* mat, Attribute& vec, Attribute& out) {
	p.params.size = vec.vboNum;
	p.params.dimention = out.dimention;
	p.params.vec = vec.vbo;
	p.params.mat = mat;
	p.block = getBlock(vec.vboNum, 1);
	p.grid = getGrid(p.block, vec.vboNum, 1, 1);

}

__global__ void TransformForwardKernel(const TransformKernelParams p) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= p.size)return;
	float3 v = ((float3*)p.vec)[pidx];
	for (int i = 0; i < p.dimention; i++) {
		p.out[pidx * p.dimention + i] = p.mat[i] * v.x + p.mat[p.dimention + i] * v.y + p.mat[p.dimention * 2 + i] * v.z + p.mat[p.dimention * 3 + i];
	}
}

void Transform::forward(TransformParams& p) {
	void* args[] = { &p.params };
	cudaLaunchKernel(TransformForwardKernel, p.grid, p.block, args, 0, NULL);
}

void Transform::init(TransformGradParams& p,  float* mat, AttributeGrad& vec, AttributeGrad& out) {
	init((TransformParams&)p,  mat, vec, out);
	p.grad.vec = vec.grad;
	p.grad.out = out.grad;
}

__global__ void TransformionBackwardKernel(const TransformKernelParams p, const TransformKernelGradParams g) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= p.size)return;
	for (int i = 0; i < p.dimention; i++) {
		g.vec[pidx * 3] += p.mat[i] * g.out[pidx * p.dimention + i];
		g.vec[pidx * 3 + 1] += p.mat[4+i] * g.out[pidx * p.dimention + i];
		g.vec[pidx * 3 + 2] += p.mat[8+i] * g.out[pidx * p.dimention + i];
	}
}

void Transform::backward(TransformGradParams& p) {
	void* args[] = { &p.params, &p.grad };
	cudaLaunchKernel(TransformionBackwardKernel, p.grid, p.block, args, 0, NULL);
}

