#pragma once
#include "common.h"
#include "buffer.h"

struct LossKernelParams{
	int width;
	int height;
	int channel;
	int depth;
	float* predict;
	float* target;
	float* grad;
};

struct LossParams{
	dim3 block;
	dim3 grid;
	LossKernelParams params;
};

class Loss {
public:
	static void init(LossParams& p, RenderBufferGrad& predict, RenderBuffer& target);
	static float loss(LossParams& p) { return 0. ;};
	static void backward(LossParams& p) {};
};

struct MSELossKernelParams {
	float* buf;
	size_t length;
};

struct MSELossParams:LossParams {
	size_t stride;
	int lh, rh;
	MSELossKernelParams mse;
};

class MSELoss :Loss {
public:
	static void init(MSELossParams& p, RenderBufferGrad& predict, RenderBuffer& target);
	static float loss(MSELossParams& p);
	static void backward(MSELossParams& p);
};



struct OptimizeKernelParams {
	int size;
	int width;
	int height;
	int channel;
	int depth;
	float* param;
	float* grad;
};

struct OptimizeParams {
	int it;
	dim3 block;
	dim3 grid;
	OptimizeKernelParams params;
};

class Optimize {
public:
	static void init(OptimizeParams& p, float* param, float* grad, int size, int width, int height, int depth, int channel);
	static void init(OptimizeParams& p, AttributeGrad& attr);
	static void init(OptimizeParams& p, MipTextureGrad& texture);
	static void addRandomParams(OptimizeParams& p, float min, float max);
	static void step(OptimizeParams& p) {};
};

struct AdamKernelParams {
	double rhom;
	double rhov;
	double rhomt;
	double rhovt;
	double eta;
	double eps;
	float* m;
	float* v;
};

struct AdamParams:OptimizeParams {
	AdamKernelParams adam;
};

class Adam {
public:
	static void init(AdamParams& p, float* param, float* grad, int size, int width, int height, int depth, int channel, double rhom, double rhov, double eta, double eps);
	static void init(AdamParams& p, AttributeGrad& attr, double rhom, double rhov, double eta, double eps);
	static void init(AdamParams& p, MipTextureGrad& texture, double rhom, double rhov, double eta, double eps);
	static void step(AdamParams& p);
};
