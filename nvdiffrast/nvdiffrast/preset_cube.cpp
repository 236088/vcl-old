#include "preset.h"

void PresetCube::Randomize() {
	Attribute::init(target_color, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::init(predict_pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::init(predict_color, target_pos.vboNum, target_pos.vaoNum, 3);
	float* target_pos_vbo, * target_color_vbo, * predict_pos_vbo, * predict_color_vbo;
	CUDA_ERROR_CHECK(cudaMallocHost(&target_pos_vbo, target_pos.vboNum * 3 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&target_color_vbo, target_pos.vboNum * 3 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&predict_color_vbo, target_pos.vboNum * 3 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(target_pos_vbo, target_pos.vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < target_pos.vboNum * 3; i++) {
		target_color_vbo[i] = (target_pos_vbo[i] + 1.f) / 2.f;
		predict_color_vbo[i] = (float)rand() / (float)RAND_MAX;
		float r = (float)rand() / (float)RAND_MAX - .5f;
		predict_pos_vbo[i] = target_pos_vbo[i] + r;
	}
	CUDA_ERROR_CHECK(cudaMemcpy(target_color.vbo, target_color_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(target_color.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(predict_pos.vbo, predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(predict_pos.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(predict_color.vbo, predict_color_vbo, target_pos.vboNum * 3 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(predict_color.vao, target_pos.vao, target_pos.vaoNum * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	CUDA_ERROR_CHECK(cudaFreeHost(target_pos_vbo));
	CUDA_ERROR_CHECK(cudaFreeHost(target_color_vbo));
	CUDA_ERROR_CHECK(cudaFreeHost(predict_pos_vbo));
	CUDA_ERROR_CHECK(cudaFreeHost(predict_color_vbo));
}

void PresetCube::init(int resolution) {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 0.f, 2.f, 5.f);
	Matrix::init(hr_mat);
	Matrix::setFovy(hr_mat, 45);
	Matrix::setEye(hr_mat, 0.f, 2.f, 5.f);
	Attribute::loadOBJ("../../cube.obj", &target_pos, nullptr, nullptr);
	Randomize();

	Project::init(predict_pp, mat.mvp, predict_pos, 4);
	Rasterize::init(predict_rp, predict_pp, predict_pos, resolution, resolution, 1, 0);
	Interpolate::init(predict_ip, predict_rp, predict_color);
	Antialias::init(predict_ap, predict_pos, predict_pp, predict_rp, predict_ip.kernel.out, 3);

	Project::init(target_pp, mat.mvp, target_pos, 4);
	Rasterize::init(target_rp, target_pp, target_pos, resolution, resolution, 1, 0);
	Interpolate::init(target_ip, target_rp, target_color);
	Antialias::init(target_ap, target_pos, target_pp, target_rp, target_ip.kernel.out, 3);

	Loss::init(loss, predict_ap.kernel.out, target_ap.kernel.out, 512, 512, 3);
	Antialias::init(predict_ap, predict_rp, loss.grad);
	Interpolate::init(predict_ip, predict_color, predict_ap.grad.in);
	Rasterize::init(predict_rp, predict_ip.grad.rast);
	Project::init(predict_pp, predict_pos, predict_rp.grad.proj);
	Adam::init(pos_adam, predict_pos, predict_pp.grad.vec, 1e-2, 0.9, 0.999, 1e-8);
	Adam::init(color_adam, predict_color, predict_ip.grad.attr, 1e-2, 0.9, 0.999, 1e-8);
	Loss::init(pos_loss, predict_pos.vbo, target_pos.vbo, target_pos.vboNum, target_pos.dimention, 1);
	Loss::init(color_loss, predict_color.vbo, target_color.vbo, target_color.vboNum, target_color.dimention, 1);

	Project::init(hr_predict_pp, hr_mat.mvp, predict_pos, 4);
	Rasterize::init(hr_predict_rp, hr_predict_pp, predict_pos, 512, 512, 1, 0);
	Interpolate::init(hr_predict_ip, hr_predict_rp, predict_color);
	Antialias::init(hr_predict_ap,predict_pos, hr_predict_pp, hr_predict_rp, hr_predict_ip.kernel.out, 3);

	Project::init(hr_target_pp, hr_mat.mvp, target_pos, 4);
	Rasterize::init(hr_target_rp, hr_target_pp, target_pos, 512, 512, 1, 0);
	Interpolate::init(hr_target_ip, hr_target_rp, target_color);
	Antialias::init(hr_target_ap, target_pos, hr_target_pp, hr_target_rp, hr_target_ip.kernel.out, 3);

	GLbuffer::init(target_buffer, target_ap.kernel.out, resolution, resolution, 3, 15);
	GLbuffer::init(predict_buffer, predict_ap.kernel.out, resolution, resolution, 3, 14);
	GLbuffer::init(hr_target_buffer, hr_target_ap.kernel.out, 512, 512, 3, 13);
	GLbuffer::init(hr_predict_buffer, hr_predict_ap.kernel.out, 512, 512, 3, 12);
}

void PresetCube::display(void) {
	Matrix::forward(mat);

	Project::forward(predict_pp);
	Rasterize::forward(predict_rp);
	Interpolate::forward(predict_ip);
	Antialias::forward(predict_ap);

	Project::forward(target_pp);
	Rasterize::forward(target_rp);
	Interpolate::forward(target_ip);
	Antialias::forward(target_ap);

	Loss::backward(loss);
	Antialias::backward(predict_ap);
	Interpolate::backward(predict_ip);
	Rasterize::backward(predict_rp);
	Project::backward(predict_pp);
	Adam::step(pos_adam);
	Adam::step(color_adam);

	Matrix::forward(hr_mat);
	Project::forward(hr_predict_pp);
	Rasterize::forward(hr_predict_rp);
	Interpolate::forward(hr_predict_ip);
	Antialias::forward(hr_predict_ap);

	Project::forward(hr_target_pp);
	Rasterize::forward(hr_target_rp);
	Interpolate::forward(hr_target_ip);
	Antialias::forward(hr_target_ap);

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