#include "preset.h"

void PresetPrimitives::init() {
	Matrix::init(mat);
	Attribute::loadOBJ("../../monkey.obj", &pos, &texel, nullptr);
	Attribute::init(proj, pos.vboNum, 4, pos.vaoNum);
	AttributeHost::init(host_proj, proj);
	RenderBuffer::init(rast, 512, 512, 4, 1);
	RenderBufferHost::init(host_rast, rast);
	RenderBuffer::init(rastDB, 512, 512, 4, 1);
	RenderBufferHost::init(host_rastDB, rastDB);
	RenderBuffer::init(intr, 512, 512, 2, 1);
	RenderBuffer::init(intrDA, 512, 512, 4, 1);
	RenderBuffer::init(tex, 512, 512, 3, 1);
	RenderBuffer::init(aa, 512, 512, 3, 1);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	MipTexture::loadBMP("../../checker.bmp", texture, 6);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	Transform::init(trans, mat.mvp, pos, proj);
	Rasterize::init(rp, rast, host_rast, rastDB, host_rastDB, proj, host_proj);
	Interpolate::init(ip, intr, intrDA, rast, rastDB, texel);
	Texturemap::init(tp, tex, texture, intr, intrDA, rast);
	Antialias::init(ap, aa, proj, tex, rast);

	GLBuffer::initGLbuffer(gl_rast, rast, 15);
	GLBuffer::initGLbuffer(gl_intr, intr, 14);
	GLBuffer::initGLbuffer(gl_tex, tex, 13);
	GLBuffer::initGLbuffer(gl_aa, aa, 12);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Transform::forward(trans);
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Texturemap::forward(tp);
	Antialias::forward(ap);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLBuffer::drawGLbuffer(gl_rast, rast, GL_RG32F, GL_RGBA, -1.0, 0.0, 0.0, 1.0);
	GLBuffer::drawGLbuffer(gl_intr, intr, GL_RG32F, GL_RG, 0.0, 1.0, 0.0, 1.0);
	GLBuffer::drawGLbuffer(gl_tex, tex, GL_RGB32F, GL_BGR, -1.0, 0.0, -1.0, 0.0);
	GLBuffer::drawGLbuffer(gl_aa, aa, GL_RGB32F, GL_BGR, 0.0, 1.0, -1.0, 0.0);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, 1.0, 0.0, 1.0, 0.0);
}