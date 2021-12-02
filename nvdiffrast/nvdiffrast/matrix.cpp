#include "matrix.h"

void Matrix::init(Matrix& mat) {
	mat.rotation = glm::mat4(1.f);
	mat.translation = glm::vec3(0.f, 0.f, 0.f);
	mat.eye = glm::vec3(0.f, 0.f, 5.f);
	mat.origin = glm::vec3(0.f, 0.f, 0.f);
	mat.up = glm::vec3(0.f, 1.f, 0.f);
	mat.fovy = 60.f;
	mat.aspect = 1.f;
	mat.znear = .1f;
	mat.zfar = 10.f;
	CUDA_ERROR_CHECK(cudaMalloc(&mat.r, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&mat.m, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&mat.mv, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&mat.mvp, 16 * sizeof(float)));
}

void Matrix::forward(Matrix& mat) {
	glm::mat4 matrix = mat.rotation;
	CUDA_ERROR_CHECK(cudaMemcpy(mat.r, &matrix, 16 * sizeof(float), cudaMemcpyHostToDevice));
	matrix = glm::translate(mat.translation) * matrix;
	CUDA_ERROR_CHECK(cudaMemcpy(mat.m, &matrix, 16 * sizeof(float), cudaMemcpyHostToDevice));
	matrix = glm::lookAt(mat.eye, mat.origin, mat.up) * matrix;
	CUDA_ERROR_CHECK(cudaMemcpy(mat.mv, &matrix, 16 * sizeof(float), cudaMemcpyHostToDevice));
	matrix = glm::perspective(glm::radians(mat.fovy), mat.aspect, mat.znear, mat.zfar) * matrix;
	CUDA_ERROR_CHECK(cudaMemcpy(mat.mvp, &matrix, 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void Matrix::setRotation(Matrix& mat, float degree, float x, float y, float z) {
	mat.rotation = glm::rotate(glm::radians(degree), glm::vec3(x, y, z));
}

void Matrix::setRandomRotation(Matrix& mat) {
	float degree = (float)rand() / (float)RAND_MAX * 3.14159265f;
	float x = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float y = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float z = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	mat.rotation = glm::rotate(mat.rotation, degree, glm::vec3(x, y, z));
}

void Matrix::addRotation(Matrix& mat, float degree, float x, float y, float z) {
	mat.rotation = glm::rotate(mat.rotation, glm::radians(degree), glm::vec3(x, y, z));
}

void Matrix::setTranslation(Matrix& mat, float x, float y, float z) {
	mat.translation = glm::vec3(x, y, z);
}

void Matrix::addTranslation(Matrix& mat, float x, float y, float z) {
	mat.translation += glm::vec3(x, y, z);
}

void Matrix::setView(Matrix& mat, float ex, float ey, float ez, float ox, float oy, float oz, float ux, float uy, float uz) {
	mat.eye = glm::vec3(ex, ey, ez);
	mat.origin = glm::vec3(ox, oy, oz);
	mat.up = glm::vec3(ux, uy, uz);
}

void Matrix::setEye(Matrix& mat, float ex, float ey, float ez) {
	mat.eye = glm::vec3(ex, ey, ez);
}

void Matrix::setOrigin(Matrix& mat, float ox, float oy, float oz) {
	mat.origin = glm::vec3(ox, oy, oz);
}

void Matrix::setProjection(Matrix& mat, float fovy, float aspect, float znear, float zfar) {
	mat.fovy = fovy;
	mat.aspect = aspect;
	mat.znear = znear;
	mat.zfar = zfar;
}

void Matrix::setVolumeZ(Matrix& mat, float znear, float zfar) {
	mat.znear = znear;
	mat.zfar = zfar;
}

void Matrix::setFovy(Matrix& mat, float fovy) {
	mat.fovy = fovy;
}