#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct LayerParams {
	int column;
	int row;

	float* W;
	float* in;
	float* rast;

	float* out;

	float* dLdout;

	float* gradW;
	float* gradIn;
};

class Layer {
public:
	static void init(LayerParams& lp, RenderingParams& p, RasterizeParams& rp, float* in, int row, int column);
	static void init(LayerParams& lp, RenderingParams& p, float* dLdout);
	static void forward(LayerParams& lp, RenderingParams& p);
	static void backward(LayerParams& lp, RenderingParams& p);
};
