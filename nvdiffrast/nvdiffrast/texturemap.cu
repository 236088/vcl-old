#include "common.h"
#include "texturemap.h"

void Texturemap::init(TexturemapParams& p, RenderBuffer& tex, MipTexture& texture, RenderBuffer& intr, RenderBuffer& intrDA, RenderBuffer& rast) {
	p.params.width = tex.width;
	p.params.height = tex.height;
	p.params.depth = tex.depth;
	p.params.texwidth = texture.width;
	p.params.texheight = texture.height;
	p.params.channel = texture.channel;
	p.params.miplevel = texture.miplevel;
	p.params.rast = rast.buffer;
	p.params.uv = intr.buffer;
	p.params.uvDA = intrDA.buffer;
	for (int i = 0; i < texture.miplevel; i++) {
		p.params.texture[i] = texture.texture[i];
	}
	p.block = getBlock(tex.width, tex.height);
	p.grid = getGrid(p.block, tex.width, tex.height, tex.depth);
}

__device__ __forceinline__ int4 indexFetch(const TexturemapKernelParams p, int level, float2 uv, float2& t) {
	int2 size = make_int2(p.texwidth >> level, p.texheight >> level);
	t.x = uv.x * (float)size.x;
	t.y = uv.y * (float)size.y;
	int u0 = t.x<0 ? 0 : t.x>size.x - 1 ? size.x - 1 : (int)t.x;
	int u1 = t.x<1 ? 0 : t.x>size.x - 2 ? size.x - 1 : (int)t.x + 1;
	int v0 = t.y<0 ? 0 : t.y>size.y - 1 ? size.y - 1 : (int)t.y;
	int v1 = t.y<1 ? 0 : t.y>size.y - 2 ? size.y - 1 : (int)t.y + 1;
	int4 idx;
	idx.x = v0 * size.x + u0;
	idx.y = v0 * size.x + u1;
	idx.z = v1 * size.x + u0;
	idx.w = v1 * size.x + u1;
	t.x = t.x<0 ? 0 : size.x<t.x ? 1 : t.x - floor(t.x);
	t.y = t.y<0 ? 0 : size.y<t.y ? 1 : t.y - floor(t.y);
	return idx;
}

__global__ void TexturemapForwardKernel(const TexturemapKernelParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.texwidth || py >= p.texheight || pz >= p.depth)return;
	int pidx = px + p.texwidth * (py + p.texheight * pz);

	if (p.rast[pidx * 4 + 3] < 1.0) return;

	float4 uvDA = ((float4*)p.uvDA)[pidx];
	float dsdx = uvDA.x * p.texwidth;
	float dsdy = uvDA.y * p.texwidth;
	float dtdx = uvDA.z * p.texheight;
	float dtdy = uvDA.w * p.texheight;

	// calculate fooprint
	// b is sum of 2 square sides 
	// b = (dsdx^2+dsdy^2) + (dtdx^2+dtdy^2)
	// c is square area
	// c = (dsdx * dtdy - dtdx * dsdy)^2
	// solve x^2 - bx + c = 0

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = 0.5 * (s2 + t2);
	float c = sqrt(b * b - a * a);

	float level = 0.5 * log2f(b + c);
	int level0 = level <= 0 ? 0 : p.miplevel - 2 <= level ? p.miplevel - 2 : (int)floor(level);
	int level1 = level <= 1 ? 1 : p.miplevel - 1 <= level ? p.miplevel - 1 : (int)floor(level) + 1;
	float flevel = level <= 0 ? 0 : p.miplevel - 1 <= level ? 1 : level - floor(level);


	float2 uv = ((float2*)p.uv)[pidx];
	float2 uv0, uv1;
	int4 idx0 = indexFetch(p, level0, uv, uv0);
	int4 idx1 = indexFetch(p, level1, uv, uv1);
	for (int i = 0; i < p.channel; i++) {
		float out = bilerp(
			p.texture[level0][idx0.x * p.channel + i], p.texture[level0][idx0.y * p.channel + i],
			p.texture[level0][idx0.z * p.channel + i], p.texture[level0][idx0.w * p.channel + i], uv0);
		if (flevel > 0) {
			float out1 = bilerp(
				p.texture[level1][idx1.x * p.channel + i], p.texture[level1][idx1.y * p.channel + i],
				p.texture[level1][idx1.z * p.channel + i], p.texture[level1][idx1.w * p.channel + i], uv1);
			out = lerp(out, out1, flevel);
		}
		p.out[pidx * p.channel + i] = out;
	}
}

void Texturemap::forward(TexturemapParams& p) {
	void* args[] = { &p.params};
	cudaLaunchKernel(TexturemapForwardKernel, p.grid, p.block, args, 0, NULL);
}

void Texturemap::init(TexturemapGradParams& p, RenderBufferGrad& tex, MipTextureGrad& texture, RenderBufferGrad& intr, RenderBufferGrad& intrDA, RenderBuffer& rast) {
	init((TexturemapParams&)p, tex, texture, intr, intrDA, rast);
	p.grad.out = tex.grad;
	p.grad.uv = intr.grad;
	p.grad.uvDA = intrDA.grad;
	for (int i = 0; i < texture.miplevel; i++) {
		p.grad.texture[i] = texture.grad[i];
	}
}

__device__ __forceinline__ void calculateLevel(const TexturemapKernelParams p, int pidx, int& level0, int& level1, float& flevel, float4& dleveldda) {
	float4 uvDA = ((float4*)p.uvDA)[pidx];
	float dsdx = uvDA.x * p.texwidth;
	float dsdy = uvDA.y * p.texwidth;
	float dtdx = uvDA.z * p.texheight;
	float dtdy = uvDA.w * p.texheight;

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = 0.5 * (s2 + t2);
	float c2 = b * b - a * a;
	float c = sqrt(c2);


	float level = 0.5 * log2f(b + c);
	level0 = level <= 0 ? 0 : p.miplevel - 2 <= level ? p.miplevel - 2 : (int)floor(level);
	level1 = level <= 1 ? 1 : p.miplevel - 1 <= level ? p.miplevel - 1 : (int)floor(level) + 1;
	flevel = level <= 0 ? 0 : p.miplevel - 1 <= level ? 1 : level - floor(level);

	float d = b * c + c2; // b^2 - a^2 == 0 or not if 0 then level=ln(b)
	if (abs(d) > 1e-6) {
		d = 0.72134752f / d;
		float bc = b + c;
		dleveldda = make_float4(d * (bc * dsdx - a * dtdy), d * (bc * dsdy + a * dtdx), (bc * dtdx + a * dsdy), (bc * dtdy - a * dsdx));
	}
	else {
		// if abs(b) == 0 then dsdx, dsdy, dtdx, dtdy are 0
		if (abs(b) > 1e-6) {
			d = 1 / b;
			dleveldda = make_float4(d * dsdx, d * dsdy, d * dtdx, d * dtdy);
		}
		else {
			dleveldda = make_float4(0.0, 0.0, 0.0, 0.0);
		}	
	}
}

// s_ = frac(s*texwidth_) => d/ds = d/ds_ * texwidth_
// t_ = frac(t*texheight_) => d/dt = d/dt_ * texheight_
// l = frac(level) => dl/dlevel = 1
//
// dL/dX = dL/dc * dc/dX
//
// dc/ds = lerp(lerp(c001-c000, c011-c010, t0) * texwidth0, lerp(c101-c100, c111-c110, t1) * texwidth1, l)
// dc/dt = lerp(lerp(c010-c000, c011-c001, s0) * texheight0, lerp(c110-c100, c111-c101, s1) * texheight1, l)
// dc/dlevel = -bilerp(c000,c001,c010,c011,s0,t0) + bilerp(c100,c101,c110,c111,s1,t1)
//
// dc/dc000 = (1-l) * (1-s0) * (1-t0)
// :
// :
// dc/dc111 = l * s1 * t1
// 
// 
//
// dL/dX = dL/dc * dc/dlevel * dlevel/dX
// 
// b = ((ds/dx^2+ds/dy^2) + (dt/dx^2+dt/dy^2))/2
// a = ds/dx * dt/dy - dt/dx * ds/dy
// level = ln(b + sqrt(b^2 - a^2))/2ln2
//
// dlevel/dX = 1/2ln2 * (b'+(b*b'-a*a')/sqrt(b^2-a^2))/(b+sqrt(b^2-a^2))
//           = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * b'- a * a')
// dlevel/d(ds/dx) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * ds/dx - a * dt/dy)
// dlevel/d(ds/dy) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * ds/dy + a * dt/dx)
// dlevel/d(dt/dx) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * dt/dx + a * ds/dy)
// dlevel/d(dt/dy) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * dt/dy - a * ds/dx)
//
__global__ void TexturemapBackwardKernel(const TexturemapKernelParams p, const TexturemapKerneGradlParams g) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= p.texwidth || py >= p.texheight || pz >= p.depth)return;
	int pidx = px + p.texwidth * (py + p.texheight * pz);
	if (p.rast[pidx * 4 + 3] < 1.0) {
		((float2*)g.uv)[pidx] = make_float2(0.0, 0.0);
		((float4*)g.uvDA)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0);
		return;
	}
	int level0 = 0, level1 = 0;
	float flevel = 0.0;
	float4 dleveldda;
	calculateLevel(p, pidx, level0, level1, flevel, dleveldda);
	float2 uv = ((float2*)p.uv)[pidx], uv0, uv1;
	float gu = 0.0, gv = 0.0, gl = 0.0;
	int4 idx0 = indexFetch(p, level0, uv, uv0);
	int4 idx1 = indexFetch(p, level1, uv, uv1);

	for (int i = 0; i < p.channel; i++) {
		float dLdout = g.out[pidx * p.channel + i];
		float check = atomicAdd(&g.texture[level0][idx0.x * p.channel + i], (1.0 - flevel) * (1.0 - uv0.x) * (1.0 - uv0.y) * dLdout);
		atomicAdd(&g.texture[level0][idx0.y * p.channel + i], (1.0 - flevel) * uv0.x * (1.0 - uv0.y) * dLdout);
		atomicAdd(&g.texture[level0][idx0.z * p.channel + i], (1.0 - flevel) * (1.0 - uv0.x) * uv0.y * dLdout);
		atomicAdd(&g.texture[level0][idx0.w * p.channel + i], (1.0 - flevel) * uv0.x * uv0.y * dLdout);
		float t00 = p.texture[level0][idx0.x * p.channel + i];
		float t01 = p.texture[level0][idx0.y * p.channel + i];
		float t10 = p.texture[level0][idx0.z * p.channel + i];
		float t11 = p.texture[level0][idx0.w * p.channel + i];

		float u = lerp(t01 - t00, t11 - t10, uv0.y) * (p.texwidth >> level0);
		float v = lerp(t10 - t00, t11 - t01, uv0.x) * (p.texheight >> level0);
		if (flevel > 0) {
			float l = bilerp(t00, t01, t10, t11, uv0);
			atomicAdd(&g.texture[level1][idx1.x * p.channel + i], flevel * (1.0 - uv1.x) * (1.0 - uv1.y) * dLdout);
			atomicAdd(&g.texture[level1][idx1.y * p.channel + i], flevel * uv1.x * (1.0 - uv1.y) * dLdout);
			atomicAdd(&g.texture[level1][idx1.z * p.channel + i], flevel * (1.0 - uv1.x) * uv1.y * dLdout);
			atomicAdd(&g.texture[level1][idx1.w * p.channel + i], flevel * uv1.x * uv1.y * dLdout);
			t00 = p.texture[level1][idx1.x * p.channel + i];
			t01 = p.texture[level1][idx1.y * p.channel + i];
			t10 = p.texture[level1][idx1.z * p.channel + i];
			t11 = p.texture[level1][idx1.w * p.channel + i];
			u = lerp(u, lerp(t01 - t00, t11 - t10, uv1.y) * (p.texwidth >> level1), flevel);
			v = lerp(v, lerp(t10 - t00, t11 - t01, uv1.x) * (p.texheight >> level1), flevel);
			gl += (bilerp(t00, t01, t10, t11, uv1) - l) * dLdout;
		}
		gu += u * dLdout;
		gv += v * dLdout;
	}

	((float2*)g.uv)[pidx] = make_float2(gu, gv);
	((float4*)g.uvDA)[pidx] = gl * dleveldda;
}

void Texturemap::backward(TexturemapGradParams& p) {
	void* args[] = { &p.params, &p.grad };
	cudaLaunchKernel(TexturemapBackwardKernel, p.grid, p.block, args, 0, NULL);
}