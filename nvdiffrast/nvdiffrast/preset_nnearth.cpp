#include "preset.h"

void PresetNNEarth::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 1.5, 1.5, 1.5);
	Rendering::init(p, 512, 512, 1);
	Attribute::loadOBJ("../../sphere.obj", pos, texel, normal);
	float* dev_params;
	cudaMallocHost(&dev_params, pos.vboNum * 8 * sizeof(float));
	Attribute::init(params, dev_params, pos.h_vao, pos.vboNum, pos.vaoNum, 8);
	cudaFreeHost(dev_params);

	Project::init(pp, mat.mvp, pos);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(target_ip, p, rp, texel);
	Texturemap::init(target_tp, p, rp, target_ip, 2048, 1536, 3, 8);
	Texturemap::loadBMP(target_tp, "../../earth-texture.bmp");
	Texturemap::buildMipTexture(target_tp);
	Antialias::init(target_ap, p, pos, pp, rp, target_tp.out, 3);
	Interpolate::init(ip, p, rp, params);
	Layer::init(lp0, p, rp, ip.out, 8, 8);
	Layer::init(lp1, p, rp, lp0.out, 8, 8);
	Layer::init(lp2, p, rp, lp1.out, 3, 8);
	Antialias::init(ap, p, pos, pp, rp, lp2.out, 3);
	Loss::init(loss, ap.out, target_ap.out, p, 3);
	Antialias::init(ap, p, rp, loss.grad);
	Layer::init(lp2, p, ap.gradIn);
	Layer::init(lp1, p, lp2.gradIn);
	Layer::init(lp0, p, lp1.gradIn);
	Interpolate::init(ip, p, params, lp0.dLdout);
	Adam::init(params_adam, params, ip.gradAttr, 1e-2,0.9, 0.999,  1e-8);
	Adam::init(adam0, lp0.W, lp0.gradW, 8 * 9, 8, 9, 1,1e-3,  0.9, 0.999, 1e-8);
	Adam::init(adam1, lp1.W, lp1.gradW, 8 * 9, 8, 9, 1, 1e-3,0.9, 0.999,  1e-8);
	Adam::init(adam2, lp2.W, lp2.gradW, 3 * 9, 3, 9, 1, 1e-3,0.9, 0.999,  1e-8);
	Optimizer::randomParams(params_adam, 0., 1.);
	Optimizer::randomParams(adam0, 0., .125);
	Optimizer::randomParams(adam1, 0., .125);
	Optimizer::randomParams(adam2, 0., .125);

	drawBufferInit(predict_buffer, p, 3, 15);
	drawBufferInit(target_buffer, p, 3, 14);
}

void PresetNNEarth::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(target_ip, p);
	Texturemap::forward(target_tp, p);
	Antialias::forward(target_ap, p);
	Interpolate::forward(ip, p);
	Layer::forward(lp0, p);
	Layer::forward(lp1, p);
	Layer::forward(lp2, p);
	Antialias::forward(ap, p);
	Loss::backward(loss);
	Antialias::backward(ap, p);
	Layer::backward(lp2, p);
	Layer::backward(lp1, p);
	Layer::backward(lp0, p);
	Interpolate::backward(ip, p);
	Adam::step(params_adam);
	Adam::step(adam0);
	Adam::step(adam1);
	Adam::step(adam2);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(target_buffer, p, target_ap.out, 3, GL_RGBA32F, GL_RGB, -1.f, 0.f, -1.f, 1.f);
	drawBuffer(predict_buffer, p, ap.out, 3, GL_RGBA32F, GL_RGB,  0.f, 1.f, -1.f, 1.f);
	glFlush();
}

void PresetNNEarth::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}