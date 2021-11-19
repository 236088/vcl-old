#include "preset.h"

void PresetEarth::init() {
	Rendering::init(p, 512, 512, 1);
	loadOBJ("../../sphere.obj", pos, texel, normal);
	Project::init(pp, pos);
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
	Texturemap::init(predict_tp, p, predict_ap.dLdout);
	Adam::init(tex_adam, predict_tp.miptex[0], predict_tp.gradMipTex[0], 2048 * 1536 * 3, 2048, 1536, 3, 0.9, 0.999, 1e-3, 1e-8);

	Project::setRotation(pp, 0.0, 0.0, 1.0, 0.0);
	Project::setView(pp, 1.5,1.5,1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	Project::setProjection(pp, 30, 1.0, 0.1, 10.0);

	Rendering::init(tex_p, 2048, 512, 1);
	drawBufferInit(predict_buffer, p, 3, 15);
	drawBufferInit(target_buffer, p, 3, 14);
	drawBufferInit(tex_target_buffer, tex_p, 3, 13);
	drawBufferInit(tex_predict_buffer, tex_p, 3, 12);
}

void PresetEarth::display(void) {
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
	Texturemap::buildMipTexture(predict_tp);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(target_buffer, p, target_tp.out, 3, GL_RGBA32F, GL_BGR, -1.0, -0.6, 0.0, 1.0);
	drawBuffer(predict_buffer, p, predict_tp.out, 3, GL_RGBA32F, GL_BGR, -1.0, -0.6, -1.0, 0.0);
	drawBuffer(tex_target_buffer, tex_p, &target_tp.miptex[0][2048 * 512 * 3], 3, GL_RGBA32F, GL_BGR, -0.6, 1.0, 0.0, 1.0);
	drawBuffer(tex_predict_buffer, tex_p, &predict_tp.miptex[0][2048 * 512 * 3], 3, GL_RGBA32F, GL_BGR, -0.6, 1.0, -1.0, 0.0);
	glFlush();
}

void PresetEarth::update(void) {
	float theta = (float)rand() / (float)RAND_MAX * 345.0;
	float x = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float y = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float z = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	Project::addRotation(pp, theta, x, y, z);
}