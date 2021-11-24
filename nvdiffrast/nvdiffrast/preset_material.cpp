#include "preset.h"

float light[32]{
	2.,2.,2.,17.,
	2.,2.,-2.,12.,
	2.,-2.,2.,15.,
	2.,-2.,-2.,10.,
	-2.,2.,2.,13.,
	-2.,2.,-2.,16.,
	-2.,-2.,2.,11.,
	-2.,-2.,-2.,14.,
};

void PresetMaterial::init() {
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 2., 1., 2.);
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
	Material::init(mp, p, rp, pos_ip, norm_ip, tp.out, 3);
	Material::init(mp,light,8, make_float3(2., 1., 2.), 0.5, 0.9);
	Antialias::init(ap, p, pos, pp, rp, mp.out, 3);

	drawBufferInit(buffer, p, 3, 15);
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
	Material::forward(mp, p);
	Antialias::forward(ap, p);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(buffer, p, ap.out, 3, GL_RGBA32F, GL_BGR, -1., 1., -1., 1.);
	glFlush();
}

void PresetMaterial::update(void) {
	Matrix::addRotation(mat, 1.0, 0.0, 1.0, 0.0);
}