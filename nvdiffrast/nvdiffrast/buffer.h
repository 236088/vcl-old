#pragma once
#include "common.h"

struct Attribute {
	float* vbo;
	unsigned int* vao;
	int vboNum;
	int vaoNum;
	int dimention;
	static void init(Attribute& attr, int vboNum, int vaoNum, int dimention);
	static void loadOBJ(const char* path, Attribute* pos, Attribute* texel, Attribute* normal);
};