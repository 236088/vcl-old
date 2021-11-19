#pragma once
#include <GL/glew.h>
#include <GL/glut.h>

#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#define __CUDACC_VER_MAJOR__ 11
#define __CUDACC_VER_MINOR__ 4
#include <device_atomic_functions.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <stdio.h>

#define MAX_DIM_PER_BLOCK 32


struct Attribute {
	float* vbo;
	float* h_vbo;
	unsigned int* vao;
	unsigned int* h_vao;
	int vboNum;
	int vaoNum;
	int dimention;
	float* grad;
};

void attributeInit(Attribute& attr, float* h_vbo, unsigned int* h_vao, int vboNum, int vaoNum, int dimention, bool learn);
void attributeGradReset(Attribute& attr);

struct RenderingParams {
	int width;
	int height;
	int depth;
	dim3 grid;
	dim3 block;
};

class Rendering {
public:
	static void init(RenderingParams& rp, int width, int height, int depth);
};

void loadOBJ(const char* path, Attribute& pos, Attribute& texel, Attribute& normal);

dim3 getBlock(int width, int height);
dim3 getGrid(dim3 block, int width, int height);

void cudaErrorCheck(const char* id, cudaError_t status);

static __device__ __forceinline__ float2& operator*=  (float2& a, const float2& b) { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ float2& operator+=  (float2& a, const float2& b) { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ float2& operator-=  (float2& a, const float2& b) { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ float2& operator*=  (float2& a, float b) { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ float2& operator+=  (float2& a, float b) { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ float2& operator-=  (float2& a, float b) { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ float2    operator*   (const float2& a, float b) { return make_float2(a.x * b, a.y * b); }
static __device__ __forceinline__ float2    operator+   (const float2& a, float b) { return make_float2(a.x + b, a.y + b); }
static __device__ __forceinline__ float2    operator-   (const float2& a, float b) { return make_float2(a.x - b, a.y - b); }
static __device__ __forceinline__ float2    operator*   (float a, const float2& b) { return make_float2(a * b.x, a * b.y); }
static __device__ __forceinline__ float2    operator+   (float a, const float2& b) { return make_float2(a + b.x, a + b.y); }
static __device__ __forceinline__ float2    operator-   (float a, const float2& b) { return make_float2(a - b.x, a - b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a) { return make_float2(-a.x, -a.y); }
static __device__ __forceinline__ float3& operator*=  (float3& a, const float3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ float3& operator+=  (float3& a, const float3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ float3& operator-=  (float3& a, const float3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ float3& operator*=  (float3& a, float b) { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ float3& operator+=  (float3& a, float b) { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ float3& operator-=  (float3& a, float b) { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ float3    operator*   (const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ float3    operator+   (const float3& a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ float3    operator-   (const float3& a, float b) { return make_float3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ float3    operator*   (float a, const float3& b) { return make_float3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ float3    operator+   (float a, const float3& b) { return make_float3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ float3    operator-   (float a, const float3& b) { return make_float3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ float4& operator*=  (float4& a, const float4& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ float4& operator+=  (float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ float4& operator-=  (float4& a, const float4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ float4& operator*=  (float4& a, float b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ float4& operator+=  (float4& a, float b) { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ float4& operator-=  (float4& a, float b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ float4    operator*   (const float4& a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ float4    operator+   (const float4& a, float b) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ float4    operator-   (const float4& a, float b) { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ float4    operator*   (float a, const float4& b) { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ float4    operator+   (float a, const float4& b) { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ float4    operator-   (float a, const float4& b) { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }

static __device__ __forceinline__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
static __device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __device__ __forceinline__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static __device__ __forceinline__ void swap(int& a, int& b) { int t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float& a, float& b) { float t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float2& a, float2& b) { float2 t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float3& a, float3& b) { float3 t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float4& a, float4& b) { float4 t = a; a = b; b = t; }
static __device__ __forceinline__ float cross(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
static __device__ __forceinline__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static __device__ __forceinline__ float lerp(float a, float b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float2 lerp2(float2 a, float2 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float4 lerp4(float4 a, float4 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float bilerp(float a, float b, float c, float d, float2 t) { return lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y); }
static __device__ __forceinline__ float2 bilerp2(float2 a, float2 b, float2 c, float2 d, float2 t) { return lerp2(lerp2(a, b, t.x), lerp2(c, d, t.x), t.y); }
static __device__ __forceinline__ float3 bilerp3(float3 a, float3 b, float3 c, float3 d, float2 t) { return lerp3(lerp3(a, b, t.x), lerp3(c, d, t.x), t.y); }
static __device__ __forceinline__ float4 bilerp4(float4 a, float4 b, float4 c, float4 d, float2 t) { return lerp4(lerp4(a, b, t.x), lerp4(c, d, t.x), t.y); }
static __device__ __forceinline__ void atomicAdd_xyw(float* ptr, float x, float y, float w) {
	atomicAdd(ptr, x);
	atomicAdd(ptr + 1, y);
	atomicAdd(ptr + 3, w);
}
static __device__ __forceinline__ void AddNaNcheck(float& a, float b) { float v = a + b; if (!isnan(v))a = v; };
