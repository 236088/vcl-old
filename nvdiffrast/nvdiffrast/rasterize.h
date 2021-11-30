#pragma once
#include "common.h"
#include "buffer.h"
#include "project.h"

struct RasterizeKernelParams {
	int width;
	int height;
	int depth;
	int enableDB;
	float xs;
	float ys;
	float* proj;
	unsigned int* idx;

	float* out;
	float* outDB;
};

struct RasterizeGradParams{
	float* proj;

	float* out;
	float* outDB;
};

struct RasterizeParams {
	RasterizeKernelParams kernel;
	RasterizeGradParams grad;
	dim3 grid;
	dim3 block;
	int enableAA = 0;

	int projNum;
	int idxNum;

	GLuint fbo;
	GLuint program;
	GLuint buffer;
	GLuint bufferDB;

	float* gl_proj;
	unsigned int* gl_idx;
	float* gl_out;
	float* gl_outDB;

};

class Rasterize {
public:
	static void init(RasterizeParams& rp, RenderingParams& p, ProjectParams& pp, Attribute& proj, int enableDB);
	static void init(RasterizeParams& rp, RenderingParams& p, float* dLdout);
	static void init(RasterizeParams& rp, RenderingParams& p, float* dLdout, float* dLddb);
	static void forward(RasterizeParams& rp);
	static void backward(RasterizeParams& rp);
};