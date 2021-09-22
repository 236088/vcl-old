#pragma once
#include "common.h"
#include "rasterize.h"

struct AntialiasParams {
	int channel;

	float* pos;
	unsigned int* idx;
	float* rast;
	float* in;
	int posNum;

	float* out;

	float* dLdout;

	float* gradPos;
	float* gradIn;
};

class Antialias {
public:
	static void forwardInit(AntialiasParams& ap, RenderingParams& p, Attribute& pos, ProjectParams& pp, RasterizeParams& rp, float* in, int channel);
	static void forward(AntialiasParams& ap, RenderingParams& p);
	static void backwardInit(AntialiasParams& ap, RenderingParams& p, float* dLdout);
	static void backward(AntialiasParams& ap, RenderingParams& p);
};
