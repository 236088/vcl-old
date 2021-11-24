#pragma once
#include "common.h"

struct ProjectParams{
	dim3 block;
	dim3 grid;

	int size;
	float* pos;
	float* mat;

	float* out;

	float* dLdout;

	float* gradPos;
	float* host_out;
};

class Project {
public:
	static void init(ProjectParams& pp, float* mat, Attribute& pos);
	static void init(ProjectParams& pp, Attribute& pos, float* dLdout);
	static void forward(ProjectParams& pp);
	static void backward(ProjectParams& pp);
};
