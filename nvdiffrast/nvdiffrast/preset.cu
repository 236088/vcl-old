#include "preset.h"

void GLbuffer::init(GLbuffer& rb, float* buffer, int width, int height, int channel, int attachmentNum) {
	rb.width = width;
	rb.height = height;
	rb.channel = channel;
	rb.buffer = buffer;
	CUDA_ERROR_CHECK(cudaMallocHost(&rb.gl_buffer, width * height * channel * sizeof(float)));
	glGenTextures(1, &rb.id);
	glBindTexture(GL_TEXTURE_2D, rb.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, rb.id, 0);
}

void GLbuffer::draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY) {
	CUDA_ERROR_CHECK(cudaMemcpy(rb.gl_buffer, rb.buffer, rb.width * rb.height * rb.channel * sizeof(float), cudaMemcpyDeviceToHost));
	glBindTexture(GL_TEXTURE_2D, rb.id);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, rb.width, rb.height, 0, format, GL_FLOAT, rb.gl_buffer);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.f, 0.f); glVertex2f(minX, minY);
	glTexCoord2f(0.f, 1.f); glVertex2f(minX, maxY);
	glTexCoord2f(1.f, 1.f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.f, 0.f); glVertex2f(maxX, minY);
	glEnd();
}