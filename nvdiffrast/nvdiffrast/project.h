#pragma once
#include "common.h"

struct ProjectParams{
	glm::mat4 transform;
	glm::vec3 eye;
	glm::vec3 origin;
	glm::vec3 up;
	glm::mat4 view;
	glm::mat4 projection;
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
	static void init(ProjectParams& pp, Attribute& pos);
	static void init(ProjectParams& pp, Attribute& pos, float* dLdout);
	static void forward(ProjectParams& pp);
	static void backward(ProjectParams& pp);
	static void setProjection(ProjectParams& pp, float fovy, float aspect, float znear, float zfar);
	static void setView(ProjectParams& pp, float ex, float ey, float ez, float ox, float oy, float oz, float ux, float uy, float uz);
	static void setRotation(ProjectParams& pp, float degree, float vx, float vy, float vz);
	static void addRotation(ProjectParams& pp, float degree, float vx, float vy, float vz);
};
