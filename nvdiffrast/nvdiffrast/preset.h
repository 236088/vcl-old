#pragma once
#include "common.h"
#include "buffer.h";
#include "matrix.h"
#include "transform.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "antialias.h"
#include "optimize.h"

struct GLBuffer {
	GLuint id;
	float* buffer;
	static void initGLbuffer(GLBuffer& gl, int width, int height, int channel, int attachmentNum);
	static void initGLbuffer(GLBuffer& gl, RenderBuffer& render, int attachmentNum);
	static void drawGLbuffer(GLBuffer& rb, float* pixels, int width, int height, int dimention, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY);
	static void drawGLbuffer(GLBuffer& rb, RenderBuffer& render, GLint internalformat, GLenum format, float minX, float maxX, float minY, float maxY);
};

class PresetPrimitives {
	Matrix mat;
	Attribute pos;
	Attribute texel;
	Attribute proj;
	AttributeHost host_proj;
	RenderBuffer rast;
	RenderBufferHost host_rast;
	RenderBuffer rastDB;
	RenderBufferHost host_rastDB;
	RenderBuffer intr;
	RenderBuffer intrDA;
	RenderBuffer tex;
	RenderBuffer aa;
	MipTexture texture;

	TransformParams trans;
	RasterizeParams rp;
	InterpolateParams ip;
	TexturemapParams tp;
	AntialiasParams ap;

	GLBuffer gl_rast;
	GLBuffer gl_intr;
	GLBuffer gl_tex;
	GLBuffer gl_aa;

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return 0.0; };
};

/*
class PresetCube {
	Matrix mat;

	Attribute target_pos;
	Attribute target_color;
	Attribute target_proj;
	TransformParams target_trans;

	AttributeGrad predict_pos;
	AttributeGrad predict_color;
	AttributeGrad predict_proj;
	TransformGradParams predict_trans;

	struct Pass {
		RenderBuffer rast;
		RenderBuffer intr;
		RenderBuffer aa;
		RasterizeParams rp;
		InterpolateParams ip;
		AntialiasParams ap;
		void init(Attribute& proj, Attribute& color);
		void forward();
	};

	RenderBufferGrad rast;
	RenderBufferGrad intr;
	RenderBufferGrad aa;
	RasterizeGradParams rp;
	InterpolateGradParams ip;
	AntialiasGradParams ap;

	Pass target;
	Pass hr_target;
	Pass hr_predict;

	GLBuffer gl_predict;
	GLBuffer gl_target;
	GLBuffer hrgl_target;
	GLBuffer hrgl_predict;

	AdamParams pos_adam;
	AdamParams color_adam;
	MSELossParams loss;

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init(int resolution);
	void display(void);
	void update(void);
	float getLoss() { return MSELoss::loss(loss); };
};

class PresetEarth {

	Attribute pos;
	Attribute texel;
	Attribute proj;

	Matrix mat;

	RenderBuffer rast;
	RenderBuffer intr;
	RenderBufferGrad tex;
	RenderBufferGrad aa;
	RenderBuffer target_tex;
	RenderBuffer target_aa;

	TransformParams trans;
	RasterizeParams rp;
	InterpolateParams p;
	TexturemapParams predict_tp;
	AntialiasParams predict_ap;

	TexturemapParams target_tp;
	AntialiasParams target_ap;

	GLBuffer gl_predict;
	GLBuffer gl_target;
	GLBuffer texgl_target;
	GLBuffer texgl_predict;

	AdamParams tex_adam;
	MSELossParams loss;

public:
	const int windowWidth = 2560;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return MSELoss::loss(loss); };
};


// original sample code 

class PresetMine {

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute proj;

	Matrix mat;

	RenderBufferGrad rast;
	RenderBufferGrad intr;
	RenderBufferGrad intr_norm;
	RenderBufferGrad intr_pos;
	RenderBufferGrad tex;
	RenderBufferGrad aa;

	TransformParams trans;
	RasterizeParams rp;
	InterpolateParams ip;
	TexturemapParams tp;
	AntialiasParams ap;

	GLBuffer buffer;

public:
	const int windowWidth = 512;
	const int windowHeight = 512;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return 0.0; };
};
*/