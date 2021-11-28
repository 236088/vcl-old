#pragma once
#include "common.h"

struct Attribute {
	float* vbo;
	float* h_vbo;
	unsigned int* vao;
	unsigned int* h_vao;
	int vboNum;
	int vaoNum;
	int dimention;
	static void init(Attribute& attr, float* h_vbo, unsigned int* h_vao, int vboNum, int vaoNum, int dimention);
	static void loadOBJ(const char* path, Attribute& pos, Attribute& texel, Attribute& normal);
	static void posShrink(Attribute& pos, float s, int repeat);
};