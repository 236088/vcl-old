#pragma once

#include "common.h"
#include "buffer.h"

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
	dim3 block;
	dim3 grid;
	InterpolateKernelParams params;
};

struct InterpolateGradParams :InterpolateParams {
	InterpolateKernelGradParams grad;
};


class Interpolate {
public:
	static void init(InterpolateParams& p, RenderBuffer& intr, RenderBuffer& rast, Attribute& attr);
	static void init(InterpolateParams& p, RenderBuffer& intr, RenderBuffer& intrDA, RenderBuffer& rast, RenderBuffer& rastDB, Attribute& attr);
	static void init(InterpolateGradParams& p, RenderBufferGrad& intr, RenderBufferGrad& rast, AttributeGrad& attr);
	static void init(InterpolateGradParams& p, RenderBufferGrad& intr, RenderBufferGrad& intrDA, RenderBufferGrad& rast, RenderBufferGrad& rastDB, AttributeGrad& attr);
	static void forward(InterpolateParams& p);
	static void backward(InterpolateGradParams& p);
};