#pragma once
#include "common.h"
#include "rasterize.h"
#include "interpolate.h"

struct MaterialParams {
	int channel;

	float* pos;
	float* normal;
	float* rast;
	float* in;

	float* out;

	float* dLdout;

	float* light;
	int lightNum;
	float3 eye;
	float roughness;
	float metallic;
};

class Material {
public:
	static void init(MaterialParams& mp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in, int channel);
	static void init(MaterialParams& mp, float* light, int lightNum, float3 eye, float roughness, float metallic);
	static void init(MaterialParams& mp, RenderingParams& p,  float* dLdout);
	static void forward(MaterialParams& mp, RenderingParams& p);
	static void backward(MaterialParams& mp, RenderingParams& p);
};

