#include "project.h"

void Project::init(ProjectParams& pp, float* mat, Attribute& pos) {
	pp.size = pos.vboNum;
	pp.block = getBlock(pos.vboNum, 1);
	pp.grid = getGrid(pp.block, pos.vboNum, 1);
	pp.pos = pos.vbo; 
	pp.mat = mat;
	cudaMallocHost(&pp.host_out, pos.vboNum * 4 * sizeof(float));
	cudaMalloc(&pp.out, pos.vboNum * 4 * sizeof(float));
}

__global__ void ProjectionForwardKernel(const ProjectParams pp) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx > pp.size)return;
	float3 v = ((float3*)pp.pos)[pidx];
	pp.out[pidx * 4] = pp.mat[0] * v.x+ pp.mat[4] * v.y + pp.mat[8] * v.z + pp.mat[12];
	pp.out[pidx * 4 + 1] = pp.mat[1] * v.x + pp.mat[5] * v.y + pp.mat[9] * v.z + pp.mat[13];
	pp.out[pidx * 4 + 2] = pp.mat[2] * v.x + pp.mat[6] * v.y + pp.mat[10] * v.z + pp.mat[14];
	pp.out[pidx * 4 + 3] = pp.mat[3] * v.x + pp.mat[7] * v.y + pp.mat[11] * v.z + pp.mat[15];
}

void Project::forward(ProjectParams& pp) {
	void* args[] = { &pp };
	cudaLaunchKernel(ProjectionForwardKernel, pp.grid, pp.block, args, 0, NULL);
	cudaMemcpy(pp.host_out, pp.out, pp.size * 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

void Project::init(ProjectParams& pp, Attribute& pos, float* dLdout) {
	pp.dLdout = dLdout;
	cudaMalloc(&pp.gradPos, pos.vboNum * pos.dimention* sizeof(float));
}

__global__ void ProjectionBackwardKernel(const ProjectParams pp) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx > pp.size)return;
	float4 v = ((float4*)pp.dLdout)[pidx];
	pp.gradPos[pidx * 3] = pp.mat[0] * v.x + pp.mat[1] * v.y + pp.mat[2] * v.z + pp.mat[3] * v.w;
	pp.gradPos[pidx * 3 + 1] = pp.mat[4] * v.x + pp.mat[5] * v.y + pp.mat[6] * v.z + pp.mat[7] * v.w;
	pp.gradPos[pidx * 3 + 2] = pp.mat[8] * v.x + pp.mat[9] * v.y + pp.mat[10] * v.z + pp.mat[11] * v.w;
}

void Project::backward(ProjectParams& pp) {
	void* args[] = { &pp };
	cudaLaunchKernel(ProjectionBackwardKernel, pp.grid, pp.block, args, 0, NULL);
}
