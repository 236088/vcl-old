#pragma once
#include "common.h"
#include "project.h"

struct RasterizeParams {
	int enableDB;
	int enableAA = 0;

	float* pos;
	unsigned int* idx;
	int posNum;
	int idxNum;

	GLuint fbo;
	GLuint program;
	GLuint buffer;
	GLuint bufferDB;

	float* out;
	float* outDB;
	float* gl_out;
	float* gl_outDB;

	float* dLdout;
	float* dLddb;
	float* gradPos;
};

class Rasterize {
public:
	static void forwardInit(RasterizeParams& rp, RenderingParams& p, ProjectParams& pp, Attribute& pos, int enableDB);
	static void forward(RasterizeParams& rp, RenderingParams& p);
	static void backwardInit(RasterizeParams& rp, RenderingParams& p, float* dLdout);
	static void backwardInit(RasterizeParams& rp, RenderingParams& p, float* dLdout, float* dLddb);
	static void backward(RasterizeParams& rp, RenderingParams& p);
};