#include "preset.h"

void GLbuffer::init(GLbuffer& rb, float* buffer, int width, int height, int channel, int attachmentNum) {
	rb.width = width;
	rb.height = height;
	rb.channel = channel;
	rb.buffer = buffer;
	cudaMallocHost(&rb.gl_buffer, width * height * channel * sizeof(float));
	glGenTextures(1, &rb.id);
	glBindTexture(GL_TEXTURE_2D, rb.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, rb.id, 0);
}

void GLbuffer::draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY) {
	cudaMemcpy(rb.gl_buffer, rb.buffer, rb.width * rb.height * rb.channel * sizeof(float), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, rb.id);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, rb.width, rb.height, 0, format, GL_FLOAT, rb.gl_buffer);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.f, 0.f); glVertex2f(minX, minY);
	glTexCoord2f(0.f, 1.f); glVertex2f(minX, maxY);
	glTexCoord2f(1.f, 1.f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.f, 0.f); glVertex2f(maxX, minY);
	glEnd();
}

__global__ void Difference(float* predict, float* target, float* diff, const RenderingParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.width || py >= p.height)return;
	int pidx = px + p.width * (py + p.height * pz);

	for (int i = 0; i < 3; i++) {
		diff[pidx * 3 + i] = predict[pidx * 3 + i]-target[pidx * 3 + i] + .5f;
	}
}

void calculateDiffrence(float* predict, float* target, float* diff, RenderingParams& p) {
	void* args[] = { &predict, &target, &diff ,&p};
	cudaLaunchKernel(Difference, p.grid, p.block, args, 0, NULL);
}