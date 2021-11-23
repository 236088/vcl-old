#include "preset.h"

/*
void PresetMine::init() {
	Rendering::init(p, 512, 512, 1);
	loadOBJ("../../monkey.obj", pos, texel, normal);
	Project::setRotation(pp, 0.0, 0.0, 1.0, 0.0);
	Project::setView(pp, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	Project::setProjection(pp, 45, 1.0, 0.1, 10.0);
	ProjectWithNormal::init(pp, pos, normal);
	Rasterize::init(rp, p, pp, pos, 1);
	Interpolate::init(ip, p, rp, normal);
	ip.attr = pp.outNorm;
	//Texturemap::init(tp, p, rp, ip, 1024, 1024, 3, 8);
	//Texturemap::loadBMP(tp, "../../checker.bmp");
	//Texturemap::buildMipTexture(tp);
	Antialias::init(ap, p, pos, pp, rp, ip.out, 3);

	drawBufferInit(buffer, p, 4, 15);
}

void PresetMine::display(void) {
	ProjectWithNormal::forward(pp);
	Rasterize::forward(rp, p);
	Interpolate::forward(ip, p);
	//Texturemap::forward(tp, p);
	Antialias::forward(ap, p);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	drawBuffer(buffer, p, ap.out, 3, GL_RGB32F, GL_RGB, -1.0, 1.0, -1.0, 1.0);
	glFlush();
}

void PresetMine::update(void) {
	Project::addRotation(pp, 1.0, 0.0, 1.0, 0.0);
}
*/