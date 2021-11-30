#pragma once
#include "common.h"
#include "buffer.h"

struct LossParams
{
	float* predict;
	float* target;
	float* grad;
	float* buffer;

	int size;
	int width;
	int height;
	int depth;
	dim3 block;
	dim3 grid;

	int stride;
	int lh;
	int rh;
};

class Loss {
public:
	static void init(LossParams& loss, float* predict, float* target, RenderingParams& p, int dimention);
	static float MSE(LossParams& loss);
	static void backward(LossParams& loss);
};


struct OptimizeParams {
	float* param;
	float* grad;
	int it;

	int size;
	int width;
	int height;
	int depth;
	dim3 block;
	dim3 grid;
};

class Optimizer {
public:
	static void randomParams(OptimizeParams& opt, float min, float max);
	static void init(OptimizeParams& opt, float* param, float* grad, int size, int width, int height, int depth);
};

struct AdamParams : OptimizeParams {
	double rhom;
	double rhov;
	double rhomt;
	double rhovt;
	double eta;
	double eps;
	float* m;
	float* v;
};


class Adam : Optimizer {
public:
	static void init(AdamParams& adam,double eta,  double rhom, double rhov, double eps);
	static void init(AdamParams& adam, float* param, float* grad, int size, int width, int height, int depth, double eta, double rhom, double rhov, double eps);
	static void init(AdamParams& adam, Attribute& attr, float* grad, double eta,  double rhom, double rhov,double eps);
	static void step(AdamParams& adam);
};

struct NadamParams : OptimizeParams {
	double mupow;
	double mupowt;
	double mu;
	double mu0;
	double mu1;
	double rho;
	double alpha;
	double mut0;
	double mut1;
	double rhot;
	double eps;
	float* m;
	float* v;
};


class Nadam : Optimizer {
public:
	static void init(NadamParams& nadam, double alpha, double mu, double rho, double eps);
	static void init(NadamParams& nadam, float* param, float* grad, int size, int width, int height, int depth,double alpha, double mu,  double rho, double eps);
	static void init(NadamParams& nadam, Attribute& attr, float* grad, double alpha,double mu,  double rho, double eps);
	static void step(NadamParams& adam);
};
