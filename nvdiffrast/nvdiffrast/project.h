#pragma once
#include "common.h"
#include "buffer.h"

struct ProjectKernelParams {
	int vboNum;
	int dimention;
	float* vbo;
	float* mat;

	float* out;
};

struct ProjectKernelGradParams {
	float* out;

	float* vbo;
};

struct ProjectParams{
	ProjectKernelParams kernel;
	ProjectKernelGradParams grad;
	dim3 block;
	dim3 grid;
	unsigned int* vao;
	int vaoNum;
};

class Project {
public:
	static void init(ProjectParams& pp, float* mat, Attribute& vec, int dimention);
	static void init(ProjectParams& pp, Attribute& vec, float* dLdout);
	static void forward(ProjectParams& pp);
	static void backward(ProjectParams& pp);
};
