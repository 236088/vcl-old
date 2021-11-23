#include "preset.h"

void PresetPrimitives::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 2.5, 2.5, 2.5);
	Rendering::init(p, 512, 512, 1);
	loadOBJ("../../monkey.obj", pos, texel, normal);
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
	drawBuffer(rp_buffer, p, rp.out, 4, GL_RG32F, GL_RGBA, -1.0, 0.0, 0.0, 1.0);
	drawBuffer(ip_buffer, p, ip.out, 2, GL_RG32F, GL_RG, 0.0, 1.0, 0.0, 1.0);
	drawBuffer(tp_buffer, p, tp.out, 3, GL_RGB32F, GL_BGR, -1.0, 0.0, -1.0, 0.0);
	drawBuffer(ap_buffer, p, ap.out, 3, GL_RGB32F, GL_BGR, 0.0, 1.0, -1.0, 0.0);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, 1.0, 0.0, 1.0, 0.0);
}