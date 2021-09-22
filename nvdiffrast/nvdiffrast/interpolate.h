#pragma once

#include "common.h"
#include "rasterize.h"

struct InterpolateParams {
	int enableDA;

	float* rast;
	float* rastDB;
	float* attr;
	unsigned int* idx;
	int attrNum;
	int idxNum;
	int dimention;

	float* out;
	float* outDA;

	float* dLdout;
	float* dLdda;

	float* gradAttr;
	float* gradRast;
	float* gradRastDB;
};

class Interpolate {
public:
	static void forwardInit(InterpolateParams& ip, RenderingParams& p, RasterizeParams& rp, Attribute& attr);
	static void forward(InterpolateParams& ip, RenderingParams& p);
	static void backwardInit(InterpolateParams& ip, RenderingParams& p, float* dLdout, float* dLdda);
	static void backwardInit(InterpolateParams& ip, RenderingParams& p, float* dLdout);
	static void backward(InterpolateParams& ip, RenderingParams& p);
};