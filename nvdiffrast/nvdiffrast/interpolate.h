#pragma once

#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct InterpolateKernelParams {
	int width;
	int height;
	int depth;
	int enableDA;
	float* rast;
	float* rastDB;
	float* attr;
	unsigned int* idx;
	int dimention;

	float* out;
	float* outDA;
};

struct InterpolateKernelGradParams {
	float* out;
	float* outDA;

	float* rast;
	float* rastDB;
	float* attr;
};

struct InterpolateParams {
	InterpolateKernelParams kernel;
	InterpolateKernelGradParams grad;
	dim3 grid;
	dim3 block;
	int attrNum;
	int idxNum;
};

class Interpolate {
public:
	static void init(InterpolateParams& ip, RasterizeParams& rp, Attribute& attr);
	static void init(InterpolateParams& ip, RasterizeParams& rp, ProjectParams& pp);
	static void init(InterpolateParams& ip, Attribute& attr, float* dLdout);
	static void init(InterpolateParams& ip, Attribute& attr, float* dLdout, float* dLdda);
	static void forward(InterpolateParams& ip);
	static void backward(InterpolateParams& ip);
};