#include "preset.h"

void PresetPrimitives::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 60);
	Matrix::setEye(mat, 1.5f, .5f, 1.5f);
	Rendering::init(p, 256, 256, 1);
	Attribute::loadOBJ("../../monkey.obj", &pos, &texel, &normal);
	Project::init(pp, mat.mvp, pos, 4);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, texel);
	Project::init(pos_pp, mat.m, pos, 3);
	Interpolate::init(pos_ip, p, rp, pos, pos_pp);
	Project::init(norm_pp, mat.r, normal, 3);
	Interpolate::init(norm_ip, p, rp, normal, norm_pp);
	Texturemap::init(tp, p, rp, ip, 1024, 1024, 3, 8);
	Texturemap::loadBMP(tp, "../../checker.bmp");
	Texturemap::buildMipTexture(tp);
	Material::init(mp, p, rp, pos_ip, norm_ip, tp.kernel.out);
	float3 lightpos[1]{
		-3.f,3.f,3.f
	};
	float3 lightintensity[1]{
		1.f,1.f,1.f
	};
	float3 ambient = make_float3(1.f, 1.f, 1.f);
	Material::init(mp, (float3*)&mat.eye, 1, lightpos, lightintensity, ambient, .2f, .6f, .8f, 4.f);
	Antialias::init(ap, p, pos, pp, rp, mp.kernel.out, 3);
	Filter::init(fp, p, ap.kernel.out, 3, 16);

	GLbuffer::init(rp_buffer, rp.kernel.out, p.width, p.height, 4, 15);
	GLbuffer::init(ip_buffer, ip.kernel.out, p.width, p.height, 2, 14);
	GLbuffer::init(pos_ip_buffer, pos_ip.kernel.out, p.width, p.height, 3, 13);
	GLbuffer::init(norm_ip_buffer, norm_ip.kernel.out, p.width, p.height, 3, 12);
	GLbuffer::init(tp_buffer, tp.kernel.out, p.width, p.height, 3, 11);
	GLbuffer::init(mp_buffer, mp.kernel.out, p.width, p.height, 3, 10);
	GLbuffer::init(ap_buffer, ap.kernel.out, p.width, p.height, 3, 9);
	GLbuffer::init(fp_buffer, fp.kernel.out, p.width, p.height, 3, 8);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(pp);
	Rasterize::forward(rp);
	Interpolate::forward(ip);
	Project::forward(pos_pp);
	Interpolate::forward(pos_ip);
	Project::forward(norm_pp);
	Interpolate::forward(norm_ip);
	Texturemap::forward(tp);
	Material::forward(mp);
	Antialias::forward(ap);
	Filter::forward(fp);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(rp_buffer, GL_RG32F, GL_RGBA, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(ip_buffer, GL_RG32F, GL_RG, -.5f, 0.f, 0.f, 1.f);
	GLbuffer::draw(pos_ip_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f);
	GLbuffer::draw(norm_ip_buffer, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	GLbuffer::draw(tp_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	GLbuffer::draw(mp_buffer, GL_RGB32F, GL_RGB, -.5f, -1.f, 0.f, 0.f);
	GLbuffer::draw(ap_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(fp_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}