#include "preset.h"

void PresetModules::drawBufferInit(RenderBuffer& rb, RenderingParams& p, int dimention, int attachmentNum) {
	cudaMallocHost(&rb.pixels, p.width * p.height * dimention * sizeof(float));
	glGenTextures(1, &rb.buffer);
	glBindTexture(GL_TEXTURE_2D, rb.buffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachmentNum, rb.buffer, 0);
}


void PresetModules::drawBuffer(RenderBuffer& rb, RenderingParams& p, float* pixels, int dimention, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY) {
	cudaMemcpy(rb.pixels, pixels, p.width * p.height * dimention * sizeof(float), cudaMemcpyDeviceToHost);
	glBindTexture(GL_TEXTURE_2D, rb.buffer);
	glTexImage2D(GL_TEXTURE_2D, 0, internalformat, p.width, p.height, 0, format, GL_FLOAT, rb.pixels);
	glBegin(GL_POLYGON);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(minX, minY);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(minX, maxY);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(maxX, maxY);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(maxX, minY);
	glEnd();
}

void PresetModules::init() {
	Rendering::init(p, 512, 512, 1);
	loadOBJ("../../monkey.obj", pos, texel, normal);
	Project::setRotation(pp, 0.0, 0.0, 1.0, 0.0);
	Project::setView(pp, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	Project::setProjection(pp, 45, 1.0, 0.1, 10.0);
	Project::forwardInit(pp, pos);
	Rasterize::forwardInit(rp, p, pp, pos, 1);
	Interpolate::forwardInit(ip, p, rp, texel);
	Texturemap::forwardInit(tp, p, rp, ip, 1024, 1024, 3, 8);
	Texturemap::loadBMP(tp, "../../checker.bmp");
	Texturemap::buildMipTexture(tp);
	Antialias::forwardInit(ap, p, pos, pp, rp, tp.out, 3);

	drawBufferInit(rp_buffer, p, 4, 15);
	drawBufferInit(ip_buffer, p, 2, 14);
	drawBufferInit(tp_buffer, p, 3, 13);
	drawBufferInit(ap_buffer, p, 3, 12);
}

void PresetModules::display(void) {
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	Texturemap::forward(tp, p);
	Antialias::forward(ap, p);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(rp_buffer, p, rp.out, 4, GL_RG32F, GL_RGBA, -1.0, 0.0, 0.0, 1.0);
	drawBuffer(ip_buffer, p, ip.out, 2, GL_RG32F, GL_RG, 0.0, 1.0, 0.0, 1.0);
	drawBuffer(tp_buffer, p, tp.out, 3, GL_RGB32F, GL_BGR, -1.0, 0.0, -1.0, 0.0);
	drawBuffer(ap_buffer, p, ap.out, 3, GL_RGB32F, GL_BGR, 0.0, 1.0, -1.0, 0.0);
	glFlush();
}

void PresetModules::update(void) {
	Project::addRotation(pp, 1.0, 0.0, 1.0, 0.0);
}