#pragma once
#include "common.h"
#include "buffer.h"

struct FilterParams {
	int channel;
	int count;

	float* filter;
	float* in;
	float* buf;

	float* out;

	float* dLdout;

	float* gradIn;
};

class Filter {
public:
	static void init(FilterParams& fp, RenderingParams& p, float* in, int channel,int count);
	static void init(FilterParams& fp, RenderingParams& p, float* dLdout);
	static void forward(FilterParams& fp, RenderingParams& p);
	static void backward(FilterParams& fp, RenderingParams& p);
};
