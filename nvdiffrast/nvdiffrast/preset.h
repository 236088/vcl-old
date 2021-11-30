#pragma once
#include "common.h"
#include "buffer.h"
#include "matrix.h"
#include "optimize.h"
#include "project.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "material.h"
#include "antialias.h"
#include "filter.h"

struct GLbuffer {
	GLuint id;
	float* gl_buffer;
	float* buffer;
	int width;
	int height;
	int channel;
	static void init(GLbuffer& rb, float* buffer, int width, int height, int channel, int attachmentNum);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY);
};

void calculateDiffrence(float* predict, float* target, float* diff, RenderingParams& p);

class PresetPrimitives {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;

	RenderingParams p;
	ProjectParams pp;
	RasterizeParams rp;
	InterpolateParams ip;
	ProjectParams pos_pp;
	InterpolateParams pos_ip;
	ProjectParams norm_pp;
	InterpolateParams norm_ip;
	TexturemapParams tp;
	MaterialParams mp;
	AntialiasParams ap;
	FilterParams fp;
	GLbuffer rp_buffer;
	GLbuffer ip_buffer;
	GLbuffer pos_ip_buffer;
	GLbuffer norm_ip_buffer;
	GLbuffer tp_buffer;
	GLbuffer mp_buffer;
	GLbuffer ap_buffer;
	GLbuffer fp_buffer;


public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
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
		void forward();
	};

	Attribute predict_pos;
	Attribute predict_color;
	Attribute target_pos;
	Attribute target_color;

	RenderingParams p;
	Pass predict;
	AdamParams pos_adam;
	AdamParams color_adam;
	LossParams loss;

	Pass target;

	RenderingParams hr_p;
	Pass hr_target;
	Pass hr_predict;

	GLbuffer predict_buffer;
	GLbuffer target_buffer;
	GLbuffer hr_target_buffer;
	GLbuffer hr_predict_buffer;


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

	RenderingParams p;

	ProjectParams pp;
	RasterizeParams rp;
	InterpolateParams ip;
	TexturemapParams predict_tp;
	AntialiasParams predict_ap;
	GLbuffer predict_buffer;

	TexturemapParams target_tp;
	AntialiasParams target_ap;
	GLbuffer target_buffer;

	GLbuffer tex_target_buffer;
	GLbuffer tex_predict_buffer;

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

// orginal preset 

class PresetMaterial {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;

	RenderingParams p;

	ProjectParams pp;
	ProjectParams norm_pp;
	ProjectParams pos_pp;
	RasterizeParams rp;
	InterpolateParams ip;
	InterpolateParams norm_ip;
	InterpolateParams pos_ip;
	TexturemapParams tp;
	MaterialParams mp;
	AntialiasParams ap;
	GLbuffer buffer;
	MaterialParams target_mp;
	AntialiasParams target_ap;
	GLbuffer target_buffer;

	AdamParams adam;
	LossParams loss;
public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return Loss::MSE(loss); };
};

class PresetFilter {
	Matrix mat;
	Matrix hr_mat;

	struct Pass {
		ProjectParams pp;
		RasterizeParams rp;
		InterpolateParams ip;
		AntialiasParams ap;
		void init(RenderingParams& p, Matrix& mat, Attribute& pos, Attribute& color);
		void forward();
	};

	Attribute predict_pos;
	Attribute predict_color;
	Attribute target_pos;
	Attribute target_color;

	RenderingParams p;
	Pass predict;
	FilterParams predict_fp;
	Pass target;
	FilterParams target_fp;

	RenderingParams hr_p;
	Pass hr_target;
	Pass hr_predict;

	GLbuffer predict_buffer;
	GLbuffer target_buffer;
	GLbuffer hr_target_buffer;
	GLbuffer hr_predict_buffer;

	AdamParams pos_adam;
	AdamParams color_adam;
	LossParams loss;

	void Randomize();
public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init(int resolution, int count);
	void display(void);
	void update(void);
	float getLoss() { return Loss::MSE(loss); };
};