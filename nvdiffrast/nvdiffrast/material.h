#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

struct MaterialKernelParams {
	int width;
	int height;
	int depth;
	float* pos;
	float* normal;
	float* rast;
	float* in;

	float* out;

	float3* eye;
	int lightNum;
	float3* lightpos;
	float3* lightintensity;
	float* params;
};

struct MaterialKernelGradParams {
	float* out;

	float* pos;
	float* normal;
	float* in;
	float3* lightpos;
	float3* lightintensity;
	float* params;
};

struct MaterialParams {
	MaterialKernelParams kernel;
	MaterialKernelGradParams grad;
	dim3 block;
	dim3 grid;
};

class Material {
public:
	static void init(MaterialParams& mp, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in);
	static void init(MaterialParams& mp, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess);
	static void init(MaterialParams& mp, float* dLdout);
	static void forward(MaterialParams& mp);
	static void backward(MaterialParams& mp);
};

