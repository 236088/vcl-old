#include "preset.h"

void GLBuffer::initGLbuffer(GLBuffer& gl, int width, int height, int channel, int attachmentNum) {
	cudaMallocHost(&gl.buffer, width * height * channel * sizeof(float));
	glGenTextures(1, &gl.id);
	glBindTexture(GL_TEXTURE_2D, gl.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, gl.id, 0);
}

void GLBuffer::initGLbuffer(GLBuffer& gl, RenderBuffer& render, int attachmentNum) {
	cudaMallocHost(&gl.buffer, render.Size());
	glGenTextures(1, &gl.id);
	glBindTexture(GL_TEXTURE_2D, gl.id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, gl.id, 0);
}

void GLBuffer::drawGLbuffer(GLBuffer& gl, float* pixels, int width, int height, int channel, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY) {
	cudaMemcpy(gl.buffer, pixels, width * height * channel * sizeof(float), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, gl.id);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, format, GL_FLOAT, gl.buffer);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(minX, minY);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(minX, maxY);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(maxX, minY);
	glEnd();
}

void GLBuffer::drawGLbuffer(GLBuffer& gl, RenderBuffer& render, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY) {
	cudaMemcpy(gl.buffer, render.buffer, render.Size(), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, gl.id);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, render.width, render.height, 0, format, GL_FLOAT, gl.buffer);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(minX, minY);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(minX, maxY);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(maxX, minY);
	glEnd();
}