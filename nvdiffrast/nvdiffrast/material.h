#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

struct MaterialParams {
	float* pos;
	float* normal;
	float* rast;
	float* in;

	float* out;

	float* dLdout;

	float* gradIn;

	float3* eye;
	int lightNum;
	float3* lightpos;
	float3* lightintensity;
	float* params;

	float3* gradLightpos;
	float3* gradLightintensity;
	float* gradParams;
};

class Material {
public:
	static void init(MaterialParams& mp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in);
	static void init(MaterialParams& mp, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess);
	static void init(MaterialParams& mp, RenderingParams& p,  float* dLdout);
	static void forward(MaterialParams& mp, RenderingParams& p);
	static void backward(MaterialParams& mp, RenderingParams& p);
};

