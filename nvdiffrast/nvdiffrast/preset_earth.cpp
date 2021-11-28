#include "preset.h"

void PresetEarth::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 30);
	Matrix::setEye(mat, 1.5, 1.5, 1.5);
	Rendering::init(p, 512, 512, 1);
	Attribute::loadOBJ("../../sphere.obj", pos, texel, normal);
	Project::init(pp, mat.mvp, pos);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, texel);
	Texturemap::init(target_tp, p, rp, ip, 2048, 1536, 3, 8);
	Texturemap::loadBMP(target_tp, "../../earth-texture.bmp");
	Texturemap::buildMipTexture(target_tp);
	Antialias::init(target_ap, p, pos, pp, rp, target_tp.out, 3);
	Texturemap::init(predict_tp, p, rp, ip, 2048, 1536, 3, 8);
	Antialias::init(predict_ap, p, pos, pp, rp, predict_tp.out, 3);
	Loss::init(loss, predict_ap.out, target_ap.out, p, 3);
	Antialias::init(predict_ap, p, rp, loss.grad);
	Texturemap::init(predict_tp, p, predict_ap.gradIn);
	Adam::init(tex_adam, predict_tp.miptex[0], predict_tp.gradMipTex[0], 2048 * 1536 * 3, 2048, 1536, 3, 1e-3, 0.9, 0.999, 1e-8);

	Rendering::init(tex_p, 2048, 512, 1);
	drawBufferInit(predict_buffer, p, 3, 15);
	drawBufferInit(target_buffer, p, 3, 14);
	drawBufferInit(tex_target_buffer, tex_p, 3, 13);
	drawBufferInit(tex_predict_buffer, tex_p, 3, 12);
}

void PresetEarth::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	Texturemap::forward(target_tp, p);
	Antialias::forward(target_ap, p);
	Texturemap::forward(predict_tp, p);
	Antialias::forward(predict_ap, p);
	Loss::backward(loss);
	Antialias::backward(predict_ap, p);
	Texturemap::backward(predict_tp, p);
	Adam::step(tex_adam);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(target_buffer, p, target_ap.out, 3, GL_RGBA32F, GL_RGB, -1.f, -0.6f, 0.f, 1.f);
	drawBuffer(predict_buffer, p, predict_ap.out, 3, GL_RGBA32F, GL_RGB, -1.f, -0.6f, -1.f, 0.f);
	drawBuffer(tex_target_buffer, tex_p, &target_tp.miptex[0][2048 * 512 * 3], 3, GL_RGBA32F, GL_RGB, -0.6f, 1.f, 0.f, 1.f);
	drawBuffer(tex_predict_buffer, tex_p, &predict_tp.miptex[0][2048 * 512 * 3], 3, GL_RGBA32F, GL_RGB, -0.6f, 1.f, -1.f, 0.f);
	glFlush();
}

void PresetEarth::update(void) {
	float theta = (float)rand() / (float)RAND_MAX * 180.f;
	float x = (float)rand() / (float)RAND_MAX *2.f - 1.f;
	float y = (float)rand() / (float)RAND_MAX *2.f - 1.f;
	float z = (float)rand() / (float)RAND_MAX *2.f - 1.f;
	Matrix::addRotation(mat, theta, x, y, z);
}