#include "preset.h"

/*
void PresetCube::Pass::init(Attribute& proj, Attribute& color) {
	RenderBuffer::init(rast, 512, 512, 4, 1);
	RenderBuffer::init(intr, 512, 512, 3, 1);
	RenderBuffer::init(aa, 512, 512, 3, 1);
	Rasterize::init(rp, rast, proj);
	Interpolate::init(ip, intr, rast,  color);
	Antialias::init(ap, aa, proj, intr, rast);
}

void PresetCube::Pass::forward() {
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Antialias::forward(ap);
}


void PresetCube::init(int resolution) {
	Rendering::init(p, resolution, resolution, 1);
	Rendering::init(hr_p, 512, 512, 1);
	loadOBJ("../../cube.obj", target_pos, texel, normal);

	float* target_color_vbo, * predict_pos_vbo, * predict_color_vbo;
	cudaMallocHost(&target_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_pos_vbo, target_pos.vboNum * 3 * sizeof(float));
	cudaMallocHost(&predict_color_vbo, target_pos.vboNum * 3 * sizeof(float));
	for (int i = 0; i < target_pos.vboNum * 3; i++) {
		target_color_vbo[i] = (target_pos.h_vbo[i] + 1.0) / 2.0;
		float r = (float)rand() / (float)RAND_MAX - 0.5;
		predict_pos_vbo[i] = target_pos.h_vbo[i] + r;
		predict_color_vbo[i] = (float)rand() / (float)RAND_MAX;
	}
	attributeInit(target_color, target_color_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3);
	attributeInit(predict_pos, predict_pos_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3);
	attributeInit(predict_color, predict_color_vbo, target_pos.h_vao, target_pos.vboNum, target_pos.vaoNum, 3);
	cudaFree(target_color_vbo); cudaFree(predict_pos_vbo); cudaFree(predict_color_vbo);

	target.init(p, target_pos, target_color);
	predict.init(p, predict_pos, predict_color);

	Loss::init(loss, predict.ap.out, target.ap.out, p, 3);
	Antialias::init(predict.ap, p, predict.rp, loss.grad);
	Interpolate::init(predict.ip, p, predict_color, predict.ap.gradIn);
	Rasterize::init(predict.rp, p, predict.ip.gradRast);
	Project::init(predict.pp, predict_pos, predict.rp.gradPos);
	Adam::init(pos_adam, predict_pos, predict_pos.vboNum, 3, 1, 0.9, 0.999, 1e-3, 1e-8);
	Adam::init(color_adam, predict_color, predict_color.vboNum, 3, 1, 0.9, 0.999, 1e-3, 1e-8);

	hr_target.init(hr_p, target_pos, target_color);
	hr_predict.init(hr_p, predict_pos, predict_color);

	drawBufferInit(target_buffer, p, 3, 15);
	drawBufferInit(predict_buffer, p, 3, 14);
	drawBufferInit(hr_target_buffer, hr_p, 3, 13);
	drawBufferInit(hr_predict_buffer, hr_p, 3, 12);
}

void PresetCube::display(void) {
	target.forward(p);
	predict.forward(p);

	Loss::backward(loss);
	attributeGradReset(predict_pos);
	attributeGradReset(predict_color);
	Antialias::backward(predict.ap, p);
	Interpolate::backward(predict.ip, p);
	Rasterize::backward(predict.rp, p);
	Project::backward(predict.pp);
	Adam::step(pos_adam);
	Adam::step(color_adam);

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
	Project::setRotation(predict.pp, theta, x, y, z);
	Project::setRotation(target.pp, theta, x, y, z);
	Project::addRotation(hr_predict.pp, 1.0, 0.0, 1.0, 0.0);
	Project::addRotation(hr_target.pp, 1.0, 0.0, 1.0, 0.0);
}
*/