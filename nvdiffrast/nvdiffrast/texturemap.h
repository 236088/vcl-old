#pragma once
#include "common.h"
#include "rasterize.h"
#include "interpolate.h"

#define TEX_MAX_MIP_LEVEL 16
#define addr(ptr, level, width, height, channel, idx) (ptr)
struct TexturemapParams {
	int width;
	int height;
	int channel;
	int miplevel;

	dim3 grid;
	dim3 block;

	float* rast;
	float* uv;
	float* uvDA;
	float* miptex[TEX_MAX_MIP_LEVEL];

	float* out;

	float* dLdout;

	float* gradUV;
	float* gradUVDA;
	float* gradMipTex[TEX_MAX_MIP_LEVEL];
};

class Texturemap {
public:
	static void init(TexturemapParams& tp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& ip, int width, int height, int channel, int miplevel);
	static void init(TexturemapParams& tp, RenderingParams& p, float* dLdout);
	static void forward(TexturemapParams& tp, RenderingParams& p);
	static void buildMipTexture(TexturemapParams& tp);
	static void backward(TexturemapParams& tp, RenderingParams& p);
	static void loadBMP(TexturemapParams& tp, const char* path);
};