#include "preset.h"


void PresetMaterial::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 2.f, 1.f, 2.f);
	Rendering::init(p, 512, 512, 1);
	Attribute::loadOBJ("../../spot_triangulated.obj", pos, texel, normal);
	Project::init(pp, mat.mvp, pos);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, texel);
	Texturemap::init(tp, p, rp, ip, 1024, 1024, 3, 8);
	Texturemap::loadBMP(tp, "../../spot_texture.bmp");
	Texturemap::buildMipTexture(tp);
	Project::init(norm_pp, mat.r, normal);
	Project::init(pos_pp, mat.m, pos);
	Interpolate::init(norm_ip, p, rp, normal, norm_pp);
	Interpolate::init(pos_ip, p, rp, pos, pos_pp);
	Material::init(target_mp, p, rp, pos_ip, norm_ip, tp.out);
	float3 lightpos[1]{
		-3.f,3.f,3.f
	};
	float3 lightintensity[1]{
		1.f,1.f,1.f
	};
	float3 ambient = make_float3(1.f, 1.f, 1.f);
	//float Ka = (float)rand() / (float)RAND_MAX;
	//float Kd = (float)rand() / (float)RAND_MAX;
	//float Ks = (float)rand() / (float)RAND_MAX;
	//float Ns = (float)rand() / (float)RAND_MAX * 9.f + 1.f;
	Material::init(target_mp, (float3*)&mat.eye, 1, lightpos, lightintensity, ambient, .1f, .5f, .7f, 3.f);
	Antialias::init(target_ap, p, pos, pp, rp, target_mp.out, 3);
	Material::init(mp, p, rp, pos_ip, norm_ip, tp.out);
	float3 zero = make_float3(0.,0.f,0.f);
	Material::init(mp, (float3*)&mat.eye, 1, lightpos, lightintensity, ambient, 0.f, 0.f, 0.f, 3.f);
	Antialias::init(ap, p, pos, pp, rp, mp.out, 3);
	Loss::init(loss, ap.out, target_ap.out, p, 3);
	Antialias::init(ap, p, rp, loss.grad);
	Material::init(mp, p, ap.gradIn);
	Adam::init(adam, mp.params, mp.gradParams, 4, 4, 1, 1, 1e-4, 0.9, 0.999, 1e-8);

	drawBufferInit(target_buffer, p, 3, 15);
	drawBufferInit(buffer, p, 3, 14);
}

void PresetMaterial::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	Texturemap::forward(tp, p);
	Project::forward(norm_pp);
	Project::forward(pos_pp);
	Interpolate::forward(norm_ip, p);
	Interpolate::forward(pos_ip, p);
	Material::forward(target_mp, p);
	Antialias::forward(target_ap, p);
	Material::forward(mp, p);
	Antialias::forward(ap, p);

	Loss::backward(loss);
	Antialias::backward(ap, p);
	Material::backward(mp, p);
	Adam::step(adam);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(target_buffer, p, target_ap.out, 3, GL_RGBA32F, GL_RGB, -1.f,0.f, -1.f, 1.f);
	drawBuffer(buffer, p, ap.out, 3, GL_RGBA32F, GL_RGB,0.f, 1.f, -1.f, 1.f);
	glFlush();
}

void PresetMaterial::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}