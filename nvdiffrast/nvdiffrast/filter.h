#pragma once
#include "common.h"
#include "buffer.h"

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

struct FilterGradParams {
	float* in;

	float* out;
};

struct FilterParams {
	FilterKernelParams kernel;
	FilterGradParams grad;
	dim3 grid;
	dim3 block;
};

class Filter {
public:
	static void init(FilterParams& fp, RenderingParams& p, float* in, int channel,int count);
	static void init(FilterParams& fp, RenderingParams& p, float* dLdout);
	static void forward(FilterParams& fp);
	static void backward(FilterParams& fp);
};
