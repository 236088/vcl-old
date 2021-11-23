#include "preset.h"

void PresetCube::Pass::init(RenderingParams& p, Matrix& mat, Attribute& pos, Attribute& color) {
	Project::init(pp, mat.mvp, pos);
	Rasterize::init(rp, p, pp, pos, 0);
	Interpolate::init(ip, p, rp, color);
	Antialias::init(ap, p, pos, pp, rp, ip.out, 3);
}

void PresetCube::Pass::forward(RenderingParams& p) {
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	Antialias::forward(ap, p);
}

void PresetCube::Randomize() {
	float* target_color_vbo, * predict_pos_vbo, * predict_color_vbo;
	cudaMallocHost(&target_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	for (int i = 0; i < target_pos.vboNum * 3; i++) {
		target_color_vbo[i] = (target_pos.h_vbo[i] + 1.0) / 2.0;
		float r = -(float)rand() / (float)RAND_MAX + 0.5;
		predict_pos_vbo[i] = target_pos.h_vbo[i] + r;
		predict_color_vbo[i] = (float)rand() / (float)RAND_MAX;
	}
	Attribute::init(target_color, target_color_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3, false);
	Attribute::init(predict_pos, predict_pos_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3, true);
	Attribute::init(predict_color, predict_color_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3, true);
	cudaFree(target_color_vbo); cudaFree(predict_pos_vbo); cudaFree(predict_color_vbo);
}

void PresetCube::init(int resolution) {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 3.0, 3.0, 2.0);
	Matrix::init(hr_mat);
	Matrix::setFovy(hr_mat, 45);
	Matrix::setEye(hr_mat, 3.0, 3.0, 2.0);
	Rendering::init(p, resolution, resolution, 1);
	Rendering::init(hr_p, 512, 512, 1);
	Attribute::loadOBJ("../../cube.obj", target_pos, texel, normal);
	Randomize();

	target.init(p, mat, target_pos, target_color);
	predict.init(p, mat, predict_pos, predict_color);

	Loss::init(loss, predict.ap.out, target.ap.out, p, 3);
	Antialias::init(predict.ap, p, predict.rp, loss.grad);
	Interpolate::init(predict.ip, p, predict_color, predict.ap.gradIn);
	Rasterize::init(predict.rp, p, predict.ip.gradRast);
	Project::init(predict.pp, predict_pos, predict.rp.gradPos);
	Adam::init(pos_adam, predict_pos, predict_pos.vboNum, 3, 1, 0.9, 0.999, 1e-3, 1e-8);
	Adam::init(color_adam, predict_color, predict_color.vboNum, 3, 1, 0.9, 0.999, 1e-3, 1e-8);

	hr_target.init(hr_p, hr_mat, target_pos, target_color);
	hr_predict.init(hr_p, hr_mat, predict_pos, predict_color);

	drawBufferInit(target_buffer, p, 3, 15);
	drawBufferInit(predict_buffer, p, 3, 14);
	drawBufferInit(hr_target_buffer, hr_p, 3, 13);
	drawBufferInit(hr_predict_buffer, hr_p, 3, 12);
}

void PresetCube::display(void) {
	Matrix::forward(mat);
	target.forward(p);
	predict.forward(p);

	Loss::backward(loss);
	Attribute::gradClear(predict_pos);
	Attribute::gradClear(predict_color);
	Antialias::backward(predict.ap, p);
	Interpolate::backward(predict.ip, p);
	Rasterize::backward(predict.rp, p);
	Project::backward(predict.pp);
	Adam::step(pos_adam);
	Adam::step(color_adam);

	Matrix::forward(hr_mat);
	hr_target.forward(hr_p);
	hr_predict.forward(hr_p);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(target_buffer, p, target.ap.out,3, GL_RGB32F, GL_RGB, 0.0, 1.0, 0.0, 1.0);
	drawBuffer(predict_buffer, p, predict.ap.out, 3, GL_RGB32F, GL_RGB, -1.0, 0.0, 0.0, 1.0);
	drawBuffer(hr_predict_buffer, hr_p, hr_target.ap.out, 3, GL_RGB32F, GL_RGB, 0.0, 1.0, -1.0, 0.0);
	drawBuffer(hr_target_buffer, hr_p, hr_predict.ap.out, 3, GL_RGB32F, GL_RGB, -1.0, 0.0, -1.0, 0.0);
	glFlush();
}

void PresetCube::update(void) {
	float theta = (float)rand() / (float)RAND_MAX * 345.0;
	float x = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float y = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	float z = (float)rand() / (float)RAND_MAX * 2.0 - 1.0;
	Matrix::setRotation(mat, theta, x, y, z);
	Matrix::addRotation(hr_mat, 1.0, 0.0, 1.0, 0.0);
}