#include "project.h"

__global__ void projectionForwardKernel(const ProjectParams pp) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx > pp.size)return;
	float3 v = ((float3*)pp.pos)[pidx];
	pp.out[pidx * 4] = pp.mat[0] * v.x+ pp.mat[4] * v.y + pp.mat[8] * v.z + pp.mat[12];
	pp.out[pidx * 4 + 1] = pp.mat[1] * v.x + pp.mat[5] * v.y + pp.mat[9] * v.z + pp.mat[13];
	pp.out[pidx * 4 + 2] = pp.mat[2] * v.x + pp.mat[6] * v.y + pp.mat[10] * v.z + pp.mat[14];
	pp.out[pidx * 4 + 3] = pp.mat[3] * v.x + pp.mat[7] * v.y + pp.mat[11] * v.z + pp.mat[15];
}

__global__ void projectionBackwardKernel(const ProjectParams pp) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx > pp.size)return;
	float4 v = ((float4*)pp.dLdout)[pidx];
	pp.gradPos[pidx * 3] = pp.mat[0] * v.x + pp.mat[1] * v.y + pp.mat[2] * v.z + pp.mat[3] * v.w;
	pp.gradPos[pidx * 3 + 1] = pp.mat[4] * v.x + pp.mat[5] * v.y + pp.mat[6] * v.z + pp.mat[7] * v.w;
	pp.gradPos[pidx * 3 + 2] = pp.mat[8] * v.x + pp.mat[9] * v.y + pp.mat[10] * v.z + pp.mat[11] * v.w;
}

void Project::forwardInit(ProjectParams& pp, Attribute& pos) {
	pp.size = pos.vboNum;
	pp.block = getBlock(pos.vboNum, 1);
	pp.grid = getGrid(pp.block, pos.vboNum, 1);
	pp.pos = pos.vbo;
	cudaMallocHost(&pp.host_out, pos.vboNum * 4 * sizeof(float));

	cudaMalloc(&pp.out, pos.vboNum * 4 * sizeof(float));
	cudaMalloc(&pp.mat, 16 * sizeof(float));
}

void Project::forward(ProjectParams& pp) {
	glm::mat4 mvp = pp.projection * pp.view * pp.transform;

	cudaMemcpy(pp.mat, &mvp, 16 * sizeof(float), cudaMemcpyHostToDevice);

	projectionForwardKernel << <pp.grid, pp.block >> > (pp);
	cudaMemcpy(pp.host_out, pp.out, pp.size * 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

void Project::backwardInit(ProjectParams& pp, Attribute& pos, float* dLdout) {
	pp.dLdout = dLdout;
	pp.gradPos = pos.grad;
}

void Project::backward(ProjectParams& pp) {
	projectionBackwardKernel << <pp.grid, pp.block >> > (pp);
}
void Project::setRotation(ProjectParams& pp, float degree, float vx, float vy, float vz) {
	pp.transform = glm::rotate(glm::radians(degree), glm::vec3(vx, vy, vz));
}

void Project::addRotation(ProjectParams& pp, float degree, float vx, float vy, float vz) {
	pp.transform = glm::rotate(pp.transform, glm::radians(degree), glm::vec3(vx, vy, vz));
}

void Project::setView(ProjectParams& pp, float ex, float ey, float ez, float ox, float oy, float oz, float ux, float uy, float uz) {
	pp.eye = glm::vec3(ex, ey, ez);
	pp.origin = glm::vec3(ox, oy, oz);
	pp.up = glm::vec3(ux, uy, uz);
	pp.view = glm::lookAt(pp.eye, pp.origin, pp.up);
}

void Project::setProjection(ProjectParams& pp, float fovy, float aspect, float znear, float zfar) {
	pp.projection = glm::perspective(glm::radians(fovy), aspect, znear, zfar);
}
