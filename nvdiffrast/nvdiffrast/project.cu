#include "project.h"

void Project::init(ProjectParams& pp, float* mat, Attribute& vec, int dimention) {
	pp.kernel.vboNum = vec.vboNum;
	pp.kernel.dimention = dimention;
	pp.kernel.vbo = vec.vbo;
	pp.kernel.mat = mat;
	pp.block = getBlock(vec.vboNum, 1);
	pp.grid = getGrid(pp.block, vec.vboNum, 1);
	CUDA_ERROR_CHECK(cudaMalloc(&pp.kernel.out, vec.vboNum * dimention * sizeof(float)));
	pp.vao = vec.vao;
	pp.vaoNum = vec.vaoNum;
}

__global__ void ProjectionForwardKernel(const ProjectKernelParams pp) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= pp.vboNum)return;
	float3 v = ((float3*)pp.vbo)[pidx];
	for (int i = 0; i < pp.dimention; i++) {
		pp.out[pidx * pp.dimention + i] = pp.mat[i] * v.x + pp.mat[4 + i] * v.y + pp.mat[8 + i] * v.z + pp.mat[12 + i];
	}
}

void Project::forward(ProjectParams& pp) {
	void* args[] = { &pp.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ProjectionForwardKernel, pp.grid, pp.block, args, 0, NULL));
}

void Project::init(ProjectParams& pp, Attribute& vec, float* dLdout) {
	pp.grad.out = dLdout;
	CUDA_ERROR_CHECK(cudaMalloc(&pp.grad.vbo, vec.vboNum * 3 * sizeof(float)));
}


__global__ void ProjectionBackwardKernel(const ProjectKernelParams pp, const ProjectKernelGradParams grad) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= pp.vboNum)return;
	for (int i = 0; i < pp.dimention; i++) {
		grad.vbo[pidx * 3] += pp.mat[i] * grad.out[pidx * pp.dimention + i];
		grad.vbo[pidx * 3 + 1] += pp.mat[4 + i] * grad.out[pidx * pp.dimention + i];
		grad.vbo[pidx * 3 + 2] += pp.mat[8 + i] * grad.out[pidx * pp.dimention + i];
	}
}

void Project::backward(ProjectParams& pp) {
	CUDA_ERROR_CHECK(cudaMemset(pp.grad.vbo, 0, pp.kernel.vboNum* 3 * sizeof(float)));
	void* args[] = { &pp.kernel,&pp.grad };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ProjectionBackwardKernel, pp.grid, pp.block, args, 0, NULL));
}
