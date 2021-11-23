#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

struct TexturemapKernelParams {
	int width;
	int height;
	int depth;
	int texwidth;
	int texheight;
	int channel;
	int miplevel;
	float* rast;

	float* uv;
	float* uvDA;
	float* texture[TEX_MAX_MIP_LEVEL];

	float* out;
};

struct TexturemapKerneGradlParams {
	float* uv;
	float* uvDA;
	float* texture[TEX_MAX_MIP_LEVEL];

	float* out;
};

struct TexturemapParams {
	dim3 grid;
	dim3 block;
	TexturemapKernelParams params;
};

struct TexturemapGradParams :TexturemapParams {
	TexturemapKerneGradlParams grad;
};

class Texturemap {
public:
	static void init(TexturemapParams& p, RenderBuffer& tex, MipTexture& texture, RenderBuffer& intr, RenderBuffer& intrDA, RenderBuffer& rast);
	static void init(TexturemapGradParams& p, RenderBufferGrad& tex, MipTextureGrad& texture, RenderBufferGrad& intr, RenderBufferGrad& intrDA, RenderBuffer& rast);
	static void forward(TexturemapParams& p);
	static void backward(TexturemapGradParams& p);
};