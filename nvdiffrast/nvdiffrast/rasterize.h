#pragma once
#include "common.h"
#include "buffer.h"

struct RasterizeKernelParams {
	int width;
	int height;
	int depth;
	int enableDB;
	float xs;
	float ys;
	float* proj;
	unsigned int* idx;

	float* out;
	float* outDB;
};

struct RasterizeKernelGradParams {
	float* proj;
	float* out;
	float* outDB;
};

struct RasterizeParams {
	GLuint fbo;
	GLuint program;
	GLuint buffer;
	GLuint bufferDB;

	float* gl_proj;
	unsigned int* gl_idx;
	float* gl_out;
	float* gl_outDB;

	size_t size;
	size_t sizeDB;
	size_t projSize;
	size_t idxSize;
	size_t idxLength;

	dim3 block;
	dim3 grid;
	RasterizeKernelParams params;
};

struct RasterizeGradParams:RasterizeParams {
	RasterizeKernelGradParams grad;
};

class Rasterize {
public:
	static void init(RasterizeParams& p, RenderBuffer& rast, RenderBufferHost& host_rast, Attribute& proj, AttributeHost& host_proj);
	static void init(RasterizeParams& p, RenderBuffer& rast, RenderBufferHost& host_rast, RenderBuffer& rastDB, RenderBufferHost& host_rastDB, Attribute& proj, AttributeHost& host_proj);
	static void init(RasterizeGradParams& p, RenderBufferGrad& rast, RenderBufferHost& host_rast, AttributeGrad& proj, AttributeHost& host_proj);
	static void init(RasterizeGradParams& p, RenderBufferGrad& rast, RenderBufferHost& host_rast, RenderBufferGrad& rastDB, RenderBufferHost& host_rastDB, AttributeGrad& proj, AttributeHost& host_proj);
	static void forward(RasterizeParams& p);
	static void backward(RasterizeGradParams& p);
};