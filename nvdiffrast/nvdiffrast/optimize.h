#pragma once
#include "common.h"

struct LossParams
{
	float* predict;
	float* correct;
	float* grad;
	float loss;

	int size;
	int width;
	int height;
	int depth;
	dim3 block;
	dim3 grid;
};

class Loss {
public:
	static void init(LossParams& loss, float* predict, float* correct, int size, int width, int height, int depth);
	static void MSE(LossParams& loss);
	static void backward(LossParams& loss);
};

struct AdamParams {
	float* param;
	float* grad;
	int it;
	double rhom;
	double rhov;
	double rhomt;
	double rhovt;
	double eta;
	double eps;
	float* m;
	float* v;

	int size;
	int width;
	int height;
	int depth;
	dim3 block;
	dim3 grid;
};

class Adam {
public:
	static void init(AdamParams& adam, float* param, float* grad, int size, int width, int height, int depth, double rhom, double rhov, double eta, double eps);
	static void step(AdamParams& adam);
};
