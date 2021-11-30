#pragma once
#include "common.h"

struct Matrix {
	glm::mat4 rotation;
	glm::vec3 translation;
	glm::vec3 eye;
	glm::vec3 origin;
	glm::vec3 up;
	float fovy;
	float aspect;
	float znear;
	float zfar;
	float* r;
	float* m;
	float* mv;
	float* mvp;
	static void init(Matrix& mat);
	static void forward(Matrix& mat);
	static void setRotation(Matrix& mat, float degree, float x, float y, float z);
	static void setRandomRotation(Matrix& mat);
	static void addRotation(Matrix& mat, float degree, float x, float y, float z);
	static void setTranslation(Matrix& mat, float x, float y, float z);
	static void addTranslation(Matrix& mat, float x, float y, float z);
	static void setView(Matrix& mat, float ex, float ey, float ez, float ox, float oy, float oz, float ux, float uy, float uz);
	static void setEye(Matrix& mat, float ex, float ey, float ez);
	static void setOrigin(Matrix& mat, float ox, float oy, float oz);
	static void setProjection(Matrix& mat, float fovy, float aspect, float znear, float zfar);
	static void setVolumeZ(Matrix& mat, float znear, float zfar);
	static void setFovy(Matrix& mat, float fovy);
};