#pragma once
#include "common.h"
#include "matrix.h"
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

void drawBufferInit(RenderBuffer& rb, RenderingParams& p, int dimention, int attachmentNum);
void drawBuffer(RenderBuffer& rb, RenderingParams& p, float* pixels, int dimention, GLint internalformat, GLenum format,  float minX, float maxX, float minY, float maxY);

class PresetPrimitives {
	Matrix mat;

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


public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return 0.0; };
};

class PresetCube {
	Matrix mat;
	Matrix hr_mat;

	struct Pass {
		ProjectParams pp;
		RasterizeParams rp;
		InterpolateParams ip;
		AntialiasParams ap;
		void init(RenderingParams& p, Matrix& mat, Attribute& pos, Attribute& color);
		void forward(RenderingParams& p);
	};

	Attribute predict_pos;
	Attribute predict_color;
	Attribute target_pos;
	Attribute target_color;
	Attribute texel;
	Attribute normal;

	RenderingParams p;
	Pass predict;
	Pass target;

	RenderingParams hr_p;
	Pass hr_target;
	Pass hr_predict;

	RenderBuffer predict_buffer;
	RenderBuffer target_buffer;
	RenderBuffer hr_target_buffer;
	RenderBuffer hr_predict_buffer;

	AdamParams pos_adam;
	AdamParams color_adam;
	LossParams loss;

	void Randomize();
public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init(int resolution);
	void display(void);
	void update(void);
	float getLoss() { return Loss::MSE(loss); };
};

class PresetEarth {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;


	RenderingParams p;

	ProjectParams pp;
	RasterizeParams rp;
	InterpolateParams ip;
	TexturemapParams predict_tp;
	AntialiasParams predict_ap;
	RenderBuffer predict_buffer;

	TexturemapParams target_tp;
	AntialiasParams target_ap;
	RenderBuffer target_buffer;

	RenderBuffer tex_target_buffer;
	RenderBuffer tex_predict_buffer;

	RenderingParams tex_p;
	AdamParams tex_adam;
	LossParams loss;

public:
	const int windowWidth = 2560;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return Loss::MSE(loss); };
};
