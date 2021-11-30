#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct AntialiasKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	float xh;
	float yh;

	float* proj;
	unsigned int* idx;
	float* rast;
	float* in;

	float* out;
};

struct AntialiasGradlParams {
	float* proj;
	float* in;

	float* out;
};

struct AntialiasParams {
	AntialiasKernelParams kernel;
	AntialiasGradlParams grad;
	dim3 block;
	dim3 grid;
	int projNum;
};

class Antialias {
public:
	static void init(AntialiasParams& ap, RenderingParams& p, Attribute& pos, ProjectParams& pp, RasterizeParams& rp, float* in, int channel);
	static void init(AntialiasParams& ap, RenderingParams& p, RasterizeParams& rp, float* dLdout);
	static void forward(AntialiasParams& ap);
	static void backward(AntialiasParams& ap);
};
