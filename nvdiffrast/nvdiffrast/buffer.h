#pragma once
#include "common.h"

struct Attribute {
	float* vbo;
	unsigned int* vao;
	int vboNum;
	int dimention;
	int vaoNum;
	size_t vboLength() { return (size_t)vboNum* dimention; };
	size_t vaoLength() { return (size_t)vaoNum* 3; };
	size_t vboSize() { return (size_t)vboNum* dimention * sizeof(float); };
	size_t vaoSize() { return (size_t)vaoNum* 3 * sizeof(unsigned); };
	static void init(Attribute& attr, int vboNum, int dimention, int vaoNum);
	static void clear(Attribute& attr);
	static void loadOBJ(const char* path, Attribute* pos, Attribute* texel, Attribute* normal);
};

struct AttributeGrad:Attribute{
	float* grad;
	static void init(AttributeGrad& attr, int vboNum, int dimention, int vaoNum);
	static void clear(AttributeGrad& attr,  bool alsoBuffer);
};

struct AttributeHost :Attribute {
	static void init(AttributeHost& attr, Attribute& src);
	static void vboMemcpyIn(AttributeHost& attr, Attribute& src);
	static void vaoMemcpyIn(AttributeHost& attr, Attribute& src);
	static void vboMemcpyOut(AttributeHost& attr, Attribute& dst);
	static void vaoMemcpyOut(AttributeHost& attr, Attribute& dst);
};

struct RenderBuffer {
	float* buffer;
	int width;
	int height;
	int channel;
	int depth;
	size_t Length() { return (size_t)width * height * channel * depth; };
	size_t Size() { return (size_t)width * height * channel * depth * sizeof(float); };
	static void init(RenderBuffer& buf, int width, int height, int channel, int depth);
	static void clear(RenderBuffer& buf);
};

struct RenderBufferGrad:RenderBuffer{
	float* grad;
	static void init(RenderBufferGrad& buf, int width, int height, int channel, int depth);
	static void clear(RenderBufferGrad& buf, bool alsoBuffer);
};

struct RenderBufferHost :RenderBuffer {
	static void init(RenderBufferHost& buf, RenderBuffer& src);
	static void MemcpyIn(RenderBufferHost& buf, RenderBuffer& src);
	static void MemcpyOut(RenderBufferHost& buf, RenderBuffer& dst);
};

struct MipTexture {
	float* texture[TEX_MAX_MIP_LEVEL];
	int width;
	int height;
	int channel;
	int miplevel;
	size_t Length(int level) { return (size_t)(width >> level) * (height >> level) * channel; };
	size_t Size(int level) { return (size_t)(width >> level) * (height >> level) * channel * sizeof(float); };
	static void init(MipTexture& miptex, int width, int height, int channel, int miplevel);
	static void clear(MipTexture& miptex);
	static void buildMIP(MipTexture& miptex);
	static void loadBMP(const char* path, MipTexture& miptex, int miplevel);
};

struct MipTextureGrad:MipTexture{
	float* grad[TEX_MAX_MIP_LEVEL];
	static void init(MipTextureGrad& miptex, int width, int height, int channel, int miplevel);
	static void clear(MipTextureGrad& miptex, bool alsoBuffer);
	static void gradSum(MipTextureGrad& miptex);
};
