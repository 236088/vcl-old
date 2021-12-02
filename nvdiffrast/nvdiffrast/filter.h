#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct FilterKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	int count;

	float* filter;
	float* in;
	float* buf;

	float* out;
};

struct FilterKernelGradParams {
	float* in;

	float* out;
};

struct FilterParams {
	FilterKernelParams kernel;
	FilterKernelGradParams grad;
	dim3 grid;
	dim3 block;
};

class Filter {
public:
	static void init(FilterParams& fp, RasterizeParams& rp, float* in, int channel,int count);
	static void init(FilterParams& fp, float* dLdout);
	static void forward(FilterParams& fp);
	static void backward(FilterParams& fp);
};
