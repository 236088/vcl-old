#pragma once
#include "common.h"
#include "optimize.h"
#include "project.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "antialias.h"

struct RenderBuffer {
	GLuint buffer;
	float* pixels;
};

class PresetModules {

	Attribute pos;
	Attribute texel;
	Attribute normal;

	RenderingParams p;
	ProjectParams pp;
	RasterizeParams rp;
	InterpolateParams ip;
	TexturemapParams tp;
	AntialiasParams ap;
	RenderBuffer rp_buffer;
	RenderBuffer ip_buffer;
	RenderBuffer tp_buffer;
	RenderBuffer ap_buffer;

	void drawBufferInit(RenderBuffer& rb, RenderingParams& p, int dimention, int attachmentNum);
	void drawBuffer(RenderBuffer& rb, RenderingParams& p, float* pixels, int dimention, GLint internalformat, GLenum format,  float minX, float maxX, float minY, float maxY);

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
};

class PresetCube {
	struct PassParams {
		ProjectParams pp;
		RasterizeParams rp;
		InterpolateParams ip;
		AntialiasParams ap;
	};

	Attribute predict_pos;
	Attribute predict_color;
	Attribute target_pos;
	Attribute target_color;
	Attribute texel;
	Attribute normal;

	RenderingParams p;
	PassParams predict;
	RenderBuffer predict_buffer;
	PassParams target;
	RenderBuffer target_buffer;

	RenderingParams hr_p;
	PassParams hr_target;
	RenderBuffer hr_target_buffer;
	PassParams hr_predict;
	RenderBuffer hr_predict_buffer;

	AdamParams pos_adam;
	AdamParams color_adam;
	LossParams loss;

	void drawBufferInit(RenderBuffer& rb, RenderingParams& p, int attachmentNum);
	void drawBuffer(RenderBuffer& rb, PassParams& pass, RenderingParams& p, float minX, float maxX, float minY, float maxY);
	void forwardInit(PassParams& pass, RenderingParams& p, Attribute& pos, Attribute& color);
	void forward(PassParams& pass, RenderingParams& p);

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init(int resolution);
	void display(void);
	void update(void);
	float getLoss() { return Loss::MSE(loss); };
};