#pragma once
#include "common.h"
#include "buffer.h"

struct TransformKernelParams{
	int size;
	int dimention;
	float* vec;
	float* mat;
	float* out;
};

struct TransformKernelGradParams{
	float* vec;
	float* out;
};

struct TransformParams {
	dim3 block;
	dim3 grid;
	TransformKernelParams params;
};

struct TransformGradParams:TransformParams {
	TransformKernelGradParams grad;
};

class Transform{
public:
	static void init(TransformParams& p, float* mat, Attribute& vec, Attribute& out);
	static void init(TransformGradParams& p, float* mat, AttributeGrad& vec, AttributeGrad& out);
	static void forward(TransformParams& p);
	static void backward(TransformGradParams& p);
};
