#include "preset.h"

void PresetPrimitives::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 2.5f, 2.5f, 2.5f);
	Rendering::init(p, 512, 512, 1);
	Attribute::loadOBJ("../../monkey.obj", pos, texel, normal);
	Project::init(pp, mat.mvp, pos);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, texel);
	Texturemap::init(tp, p, rp, ip, 1024, 1024, 3, 8);
	Texturemap::loadBMP(tp, "../../checker.bmp");
	Texturemap::buildMipTexture(tp);
	Antialias::init(ap, p, pos, pp, rp, tp.out, 3);

	drawBufferInit(rp_buffer, p, 4, 15);
	drawBufferInit(ip_buffer, p, 2, 14);
	drawBufferInit(tp_buffer, p, 3, 13);
	drawBufferInit(ap_buffer, p, 3, 12);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	Texturemap::forward(tp, p);
	Antialias::forward(ap, p);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(rp_buffer, p, rp.out, 4, GL_RG32F, GL_RGBA, -1.f, 0.f, 0.f, 1.f);
	drawBuffer(ip_buffer, p, ip.out, 2, GL_RG32F, GL_RG, 0.f, 1.f, 0.f, 1.f);
	drawBuffer(tp_buffer, p, tp.out, 3, GL_RGB32F, GL_RGB, -1.f, 0.f, -1.f, 0.f);
	drawBuffer(ap_buffer, p, ap.out, 3, GL_RGB32F, GL_RGB, 0.f, 1.f, -1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}