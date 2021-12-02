#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

#define TEX_MAX_MIP_LEVEL 16

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

struct TexturemapKernelGradParams {
	float* out;

	float* uv;
	float* uvDA;
	float* texture[TEX_MAX_MIP_LEVEL];
};

struct TexturemapParams {
	TexturemapKernelParams kernel;
	TexturemapKernelGradParams grad;
	dim3 grid;
	dim3 block;
	dim3 texgrid;
	dim3 texblock;
};

class Texturemap {
public:
	static void init(TexturemapParams& tp, RasterizeParams& rp, InterpolateParams& ip, int texwidth, int texheight, int channel, int miplevel);
	static void init(TexturemapParams& tp, float* dLdout);
	static void forward(TexturemapParams& tp);
	static void backward(TexturemapParams& tp);
	static void buildMipTexture(TexturemapParams& tp);
	static void loadBMP(TexturemapParams& tp, const char* path);
};