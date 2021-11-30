#include "preset.h"

void PresetCube::Pass::init(RenderingParams& p, Matrix& mat, Attribute& pos, Attribute& color) {
	Project::init(pp, mat.mvp, pos, 4);
	Rasterize::init(rp, p, pp, pos, 0);
	Interpolate::init(ip, p, rp, color);
	Antialias::init(ap, p, pos, pp, rp, ip.kernel.out, 3);
}

void PresetCube::Pass::forward() {
	Project::forward(pp);
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Antialias::forward(ap);
}

void PresetCube::Randomize() {
	Attribute::init(target_color, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::init(predict_pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::init(predict_color, target_pos.vboNum, target_pos.vaoNum, 3);
	float* target_pos_vbo,* target_color_vbo, * predict_pos_vbo, * predict_color_vbo;
	cudaMallocHost(&target_pos_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&target_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMemcpy(target_pos_vbo, target_pos.vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < target_pos.vboNum * 3; i++) {
		target_color_vbo[i] = (target_pos_vbo[i] + 1.f) / 2.f;
		predict_color_vbo[i] = (float)rand() / (float)RAND_MAX;
		float r = (float)rand() / (float)RAND_MAX - .5f;
		predict_pos_vbo[i] = target_pos_vbo[i] + r;
	}
	cudaMemcpy(target_color.vbo, target_color_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(target_color.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(predict_pos.vbo, predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(predict_pos.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(predict_color.vbo, predict_color_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(predict_color.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaFree(target_pos_vbo); cudaFree(target_color_vbo); cudaFree(predict_pos_vbo); cudaFree(predict_color_vbo);
}

void PresetCube::init(int resolution) {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 0.f, 2.f, 5.f);
	Matrix::init(hr_mat);
	Matrix::setFovy(hr_mat, 45);
	Matrix::setEye(hr_mat, 0.f, 2.f, 5.f);
	Rendering::init(p, resolution, resolution, 1);
	Rendering::init(hr_p, 512, 512, 1);
	Attribute::loadOBJ("../../cube.obj", &target_pos, nullptr, nullptr);
	Randomize();

	target.init(p, mat, target_pos, target_color);
	predict.init(p, mat, predict_pos, predict_color);

	Loss::init(loss, predict.ap.kernel.out, target.ap.kernel.out, p, 3);
	Antialias::init(predict.ap, p, predict.rp, loss.grad);
	Interpolate::init(predict.ip, p, predict_color, predict.ap.grad.in);
	Rasterize::init(predict.rp, p, predict.ip.grad.rast);
	Project::init(predict.pp, predict_pos, predict.rp.grad.proj);
	Adam::init(pos_adam, predict_pos, predict.pp.grad.vec, 1e-2, 0.9, 0.999, 1e-8);
	Adam::init(color_adam, predict_color, predict.ip.grad.attr, 1e-2, 0.9, 0.999, 1e-8);

	hr_target.init(hr_p, hr_mat, target_pos, target_color);
	hr_predict.init(hr_p, hr_mat, predict_pos, predict_color);

	GLbuffer::init(target_buffer, target.ap.kernel.out, p.width, p.height, 3, 15);
	GLbuffer::init(predict_buffer, predict.ap.kernel.out, p.width, p.height, 3, 14);
	GLbuffer::init(hr_target_buffer, hr_target.ap.kernel.out, hr_p.width, hr_p.height, 3, 13);
	GLbuffer::init(hr_predict_buffer, hr_predict.ap.kernel.out, hr_p.width, hr_p.height, 3, 12);
}

void PresetCube::display(void) {
	Matrix::forward(mat);
	target.forward();
	predict.forward();

	Loss::backward(loss);
	Antialias::backward(predict.ap);
	Interpolate::backward(predict.ip);
	Rasterize::backward(predict.rp);
	Project::backward(predict.pp);
	Adam::step(pos_adam);
	Adam::step(color_adam);

	Matrix::forward(hr_mat);
	hr_target.forward();
	hr_predict.forward();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(predict_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, 1.f, 1.f);
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
	GLbuffer::draw(hr_predict_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f,1.f,  0.f);
	GLbuffer::draw(hr_target_buffer,  GL_RGB32F, GL_RGB, -1.f,  -1.f, 0.f,0.f);
	glFlush();
}

void PresetCube::update(void) {
	Matrix::setRandomRotation(mat);
	Matrix::addRotation(hr_mat, 1.f, 0.f, 1.f, 0.f);
}