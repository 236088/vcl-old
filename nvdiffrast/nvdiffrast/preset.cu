#include "preset.h"

void drawBufferInit(RenderBuffer& rb, RenderingParams& p, int dimention, int attachmentNum) {
	cudaMallocHost(&rb.pixels, p.width * p.height * dimention * sizeof(float));
	glGenTextures(1, &rb.buffer);
	glBindTexture(GL_TEXTURE_2D, rb.buffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, rb.buffer, 0);
}


void drawBuffer(RenderBuffer& rb, RenderingParams& p, float* pixels, int dimention, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY) {
	cudaMemcpy(rb.pixels, pixels, p.width * p.height * dimention * sizeof(float), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, rb.buffer);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, p.width, p.height, 0, format, GL_FLOAT, rb.pixels);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.f, 0.f); glVertex2f(minX, minY);
	glTexCoord2f(0.f, 1.f); glVertex2f(minX, maxY);
	glTexCoord2f(1.f, 1.f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.f, 0.f); glVertex2f(maxX, minY);
	glEnd();
}