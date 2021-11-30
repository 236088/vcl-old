#pragma once
#include "common.h"
#include "buffer.h"

struct ProjectKernelParams {
	int size;
	int dimention;
	float* vec;
	float* mat;

	float* out;
};

struct ProjectGradParams {
	float* out;

	float* vec;
};

struct ProjectParams{
	ProjectKernelParams kernel;
	ProjectGradParams grad;
	dim3 block;
	dim3 grid;
};

class Project {
public:
	static void init(ProjectParams& pp, float* mat, Attribute& vec, int dimention);
	static void init(ProjectParams& pp, Attribute& vec, float* dLdout);
	static void forward(ProjectParams& pp);
	static void backward(ProjectParams& pp);
};
