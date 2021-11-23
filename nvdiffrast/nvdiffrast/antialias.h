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
	float* rast;
	unsigned int* idx;

	float* proj;
	float* in;

	float* out;
};

struct AntialiasKernelGradParams {
	float* proj;
	float* in;

	float* out;
};

struct AntialiasParams {
	dim3 grid;
	dim3 block;
	AntialiasKernelParams params;
};

struct AntialiasGradParams :AntialiasParams {
	AntialiasKernelGradParams grad;
};

class Antialias {
public:
	static void init(AntialiasParams& p, RenderBuffer& aa, Attribute& proj, RenderBuffer& in, RenderBuffer& rast);
	static void init(AntialiasGradParams& p, RenderBufferGrad& aa, AttributeGrad& proj, RenderBufferGrad& in, RenderBuffer& rast);
	static void forward(AntialiasParams& p);
	static void backward(AntialiasGradParams& p);
};
