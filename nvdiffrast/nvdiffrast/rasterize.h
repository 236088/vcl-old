#pragma once
#include "common.h"
#include "buffer.h"
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
	static void init(RasterizeParams& rp, RenderingParams& p, ProjectParams& pp, Attribute& pos, int enableDB);
	static void init(RasterizeParams& rp, RenderingParams& p, float* dLdout);
	static void init(RasterizeParams& rp, RenderingParams& p, float* dLdout, float* dLddb);
	static void forward(RasterizeParams& rp, RenderingParams& p);
	static void backward(RasterizeParams& rp, RenderingParams& p);
};