#include "preset.h"

void PresetEarth::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 30);
	Matrix::setEye(mat, 1.5, 1.5, 1.5);
	Rendering::init(p, 512, 512, 1);
	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	Project::init(pp, mat.mvp, pos, 4);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, texel);
	Texturemap::init(target_tp, p, rp, ip,2048, 1536, 3, 8);
	Texturemap::loadBMP(target_tp, "../../earth-texture.bmp");
	Texturemap::buildMipTexture(target_tp);
	Antialias::init(target_ap, p, pos, pp, rp, target_tp.kernel.out, 3);
	Texturemap::init(predict_tp, p, rp, ip,2048, 1536, 3, 4);
	Antialias::init(predict_ap, p, pos, pp, rp, predict_tp.kernel.out, 3);
	Loss::init(loss, predict_ap.kernel.out, target_ap.kernel.out, p, 3);
	Antialias::init(predict_ap, p, rp, loss.grad);
	Texturemap::init(predict_tp, p, predict_ap.grad.in);
	Adam::init(tex_adam, predict_tp.kernel.texture[0], predict_tp.grad.texture[0],2048 * 1536 * 3,2048, 1536, 3, 1e-3, 0.9, 0.999, 1e-8);

	Rendering::init(tex_p, 2048, 512, 1);
	GLbuffer::init(predict_buffer, predict_ap.kernel.out, p.width, p.height, 3, 15);
	GLbuffer::init(target_buffer, target_ap.kernel.out, p.width, p.height, 3, 14);
	GLbuffer::init(tex_target_buffer, &target_tp.kernel.texture[0][2048 * 512 * 3], 2048, 512, 3, 13);
	GLbuffer::init(tex_predict_buffer, &predict_tp.kernel.texture[0][2048 * 512 * 3], 2048, 512, 3, 12);
}

void PresetEarth::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Texturemap::forward(target_tp);
	Antialias::forward(target_ap);
	Texturemap::forward(predict_tp);
	Antialias::forward(predict_ap);
	Loss::backward(loss);
	Antialias::backward(predict_ap);
	Texturemap::backward(predict_tp);
	Adam::step(tex_adam);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGBA32F, GL_RGB, -1.f, 0.f, -0.6f, 1.f);
	GLbuffer::draw(predict_buffer, GL_RGBA32F, GL_RGB, -1.f, -1.f, -0.6f, 0.f);
	GLbuffer::draw(tex_target_buffer, GL_RGBA32F, GL_RGB, -0.6f, 0.f, 1.f, 1.f);
	GLbuffer::draw(tex_predict_buffer, GL_RGBA32F, GL_RGB, -0.6f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetEarth::update(void) {
	Matrix::setRandomRotation(mat);
}