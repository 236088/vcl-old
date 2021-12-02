#include "preset.h"


void PresetMaterial::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 2.f, 1.f, 2.f);
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, &normal);
	Project::init(pp, mat.mvp, pos, 4);
	Rasterize::init(rp, pp, pos, 512, 512, 1, 1);
	Interpolate::init(ip, rp, texel);
	Texturemap::init(tp, rp, ip, 1024, 1024, 3, 8);
	Texturemap::loadBMP(tp, "../../spot_texture.bmp");
	Texturemap::buildMipTexture(tp);
	Project::init(norm_pp, mat.r, normal,3);
	Project::init(pos_pp, mat.m, pos, 3);
	Interpolate::init(norm_ip, rp, normal, norm_pp);
	Interpolate::init(pos_ip, rp, pos, pos_pp);
	Material::init(target_mp, rp, pos_ip, norm_ip, tp.kernel.out);
	float3 lightpos[1]{
		-3.f,3.f,3.f
	};
	float3 lightintensity[1]{
		1.f,1.f,1.f
	};
	float3 ambient = make_float3(1.f, 1.f, 1.f);
	float Ks = (float)rand() / (float)RAND_MAX * .5f + .5f;
	float Kd = (float)rand() / (float)RAND_MAX * .5f + .5f;
	float Ka = (float)rand() / (float)RAND_MAX * .5f;
	float shininess = (float)rand() / (float)RAND_MAX * 10.f + 1.f;;
	Material::init(target_mp, (float3*)&mat.eye, 1, lightpos, lightintensity, ambient, Ka, Kd, Ks, shininess);
	Antialias::init(target_ap, pos, pp, rp, target_mp.kernel.out, 3);
	Material::init(mp, rp, pos_ip, norm_ip, tp.kernel.out);
	float3 zero = make_float3(0.,0.f,0.f);
	Material::init(mp, (float3*)&mat.eye, 1, lightpos, lightintensity, ambient, 0.f, 0.f, 0.f, 3.f);
	Antialias::init(ap, pos, pp, rp, mp.kernel.out, 3);
	Loss::init(loss, ap.kernel.out, target_ap.kernel.out, 512, 512, 3);
	Antialias::init(ap, rp, loss.grad);
	Material::init(mp, ap.grad.in);
	Adam::init(adam, mp.kernel.params, mp.grad.params, 4, 4, 1, 1, 1e-3, 0.9, 0.999, 1e-8);

	GLbuffer::init(target_buffer, target_ap.kernel.out, 512, 512, 3, 15);
	GLbuffer::init(buffer, ap.kernel.out, 512, 512, 3, 14);
}

void PresetMaterial::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Texturemap::forward(tp);
	Project::forward(norm_pp);
	Project::forward(pos_pp);
	Interpolate::forward(norm_ip);
	Interpolate::forward(pos_ip);
	Material::forward(target_mp);
	Antialias::forward(target_ap);
	Material::forward(mp);
	Antialias::forward(ap);

	Loss::backward(loss);
	Antialias::backward(ap);
	Material::backward(mp);
	Adam::step(adam);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGBA32F, GL_RGB, -1.f, -1.f, 0.f, 1.f);
	GLbuffer::draw(buffer, GL_RGBA32F, GL_RGB, 0.f, -1.f, 1.f, 1.f);
	glFlush();
}

void PresetMaterial::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}