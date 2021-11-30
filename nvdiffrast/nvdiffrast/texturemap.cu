#include "common.h"
#include "texturemap.h"

void Texturemap::init(TexturemapParams& tp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& ip, int texwidth, int texheight, int channel, int miplevel) {
	miplevel = miplevel < TEX_MAX_MIP_LEVEL ? miplevel : TEX_MAX_MIP_LEVEL;
	if(((texwidth >> miplevel) << miplevel) != texwidth || ((texheight >> miplevel) << miplevel) != texheight){
		printf("Invalid miplevel value");
		exit(1);
	}
	tp.kernel.width = p.width;
	tp.kernel.height = p.height;
	tp.kernel.depth = p.depth;
	tp.kernel.texwidth = texwidth;
	tp.kernel.texheight = texheight;
	tp.kernel.channel = channel;
	tp.kernel.miplevel = miplevel;
	tp.kernel.rast = rp.kernel.out;
	tp.kernel.uv = ip.kernel.out;
	tp.kernel.uvDA = ip.kernel.outDA;
	tp.texblock = getBlock(texwidth, texheight);
	tp.texgrid = getGrid(tp.texblock, texwidth, texheight);
	tp.block = getBlock(p.width, p.height);
	tp.grid = getGrid(tp.block, p.width, p.height);

	int w = texwidth, h = texheight;
	for (int i = 0; i < miplevel; i++) {
		cudaMalloc(&tp.kernel.texture[i], w * h * channel * sizeof(float));
		w >>= 1; h >>= 1;
	}
	cudaMalloc(&tp.kernel.out, p.width * p.height * channel * sizeof(float));
}

__global__ void bmpUcharToFloat(unsigned char* data, const TexturemapKernelParams tp) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tp.texwidth || py >= tp.texheight)return;
	int pidx = px + tp.texwidth * (py + tp.texheight * pz);

	for (int i = 0; i < tp.channel; i++) {
		tp.texture[0][pidx * tp.channel + i] = (float)data[pidx * tp.channel + 2 - i] / 255.f;
	}
}

void Texturemap::loadBMP(TexturemapParams& tp, const char* path) {
	unsigned char header[54];

	FILE* file = fopen(path, "rb");
	if (!file) {
		printf("Image could not be opened\n");
		return;
	}
	if (fread(header, 1, 54, file) != 54) {
		printf("Not a correct BMP file\n");
		return;
	}
	if (header[0] != 'B' || header[1] != 'M') {
		printf("Not a correct BMP file\n");
		return;
	}
	if (*(int*)&(header[0x12]) != tp.kernel.texwidth || *(int*)&(header[0x16]) != tp.kernel.texheight) {
		printf("Not match texWidth or texHeight value\n");
		return;
	}
	unsigned int dataPos = *(int*)&(header[0x0A]);
	unsigned int imageSize = *(int*)&(header[0x22]);

	if (imageSize == 0)    imageSize = tp.kernel.texwidth * tp.kernel.texheight * tp.kernel.channel;
	if (dataPos == 0)      dataPos = 54;
	fseek(file, dataPos, SEEK_SET);

	unsigned char* data = new unsigned char[imageSize];
	fread(data, 1, imageSize, file);
	fclose(file);

	unsigned char* dev_data;

	cudaMalloc(&dev_data, tp.kernel.texwidth * tp.kernel.texheight * tp.kernel.channel * sizeof(unsigned char));
	cudaMemcpy(dev_data, data, tp.kernel.texwidth * tp.kernel.texheight * tp.kernel.channel * sizeof(unsigned char), cudaMemcpyHostToDevice);

	void* args[] = { &dev_data,&tp.kernel };
	cudaLaunchKernel(bmpUcharToFloat, tp.texgrid, tp.texblock, args, 0, NULL);
	cudaFree(dev_data);
}

__global__ void downSampling(const TexturemapKernelParams tp, int index, int width, int height) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= width || py >= height)return;
	int pidx = px + width * (py + height * pz);
	px <<= 1; py <<= 1;
	width <<= 1; height <<= 1;
	int p00idx = px + width * (py + height * pz);
	int p01idx = p00idx + 1;
	int p10idx = p00idx + width;
	int p11idx = p10idx + 1;

	for (int i = 0; i < tp.channel; i++) {
		float p00 = tp.texture[index - 1][p00idx * tp.channel + i];
		float p01 = tp.texture[index - 1][p01idx * tp.channel + i];
		float p10 = tp.texture[index - 1][p10idx * tp.channel + i];
		float p11 = tp.texture[index - 1][p11idx * tp.channel + i];

		float p = (p00 + p01 + p10 + p11) * 0.25f;
		tp.texture[index][pidx * tp.channel + i] = p;
	}
}

void Texturemap::buildMipTexture(TexturemapParams& tp) {
	int w = tp.kernel.texwidth, h = tp.kernel.texheight;
	int i = 0;
	void* args[] = { &tp.kernel, &i, &w, &h };
	for (i = 1; i < tp.kernel.miplevel; i++) {
		w >>= 1; h >>= 1;
		dim3 block = getBlock(w, h);
		dim3 grid = getGrid(block, w, h);
		cudaLaunchKernel(downSampling, grid, block, args, 0, NULL);
	}
}

__device__ __forceinline__ int4 indexFetch(const TexturemapKernelParams tp, int level, float2 uv, float2& t) {
	int2 size = make_int2(tp.texwidth >> level, tp.texheight >> level);
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

__global__ void TexturemapForwardKernel(const TexturemapKernelParams tp) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tp.width || py >= tp.height || pz >= tp.depth)return;
	int pidx = px + tp.width * (py + tp.height * pz);

	if (tp.rast[pidx * 4 + 3] < 1.f) return;

	float4 uvDA = ((float4*)tp.uvDA)[pidx];
	float dsdx = uvDA.x * tp.texwidth;
	float dsdy = uvDA.y * tp.texwidth;
	float dtdx = uvDA.z * tp.texheight;
	float dtdy = uvDA.w * tp.texheight;

	// calculate footprint
	// b is sum of 2 square sides 
	// b = (dsdx^2+dsdy^2) + (dtdx^2+dtdy^2)
	// c is square area
	// c = (dsdx * dtdy - dtdx * dsdy)^2
	// solve x^2 - bx + c = 0

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = .5f * (s2 + t2);
	float c = sqrt(b * b - a * a);

	float level = .5f * log2f(b + c);
	int level0 = level <= 0 ? 0 : tp.miplevel - 2 <= level ? tp.miplevel - 2 : (int)floor(level);
	int level1 = level <= 1 ? 1 : tp.miplevel - 1 <= level ? tp.miplevel - 1 : (int)floor(level) + 1;
	float flevel = level <= 0 ? 0 : tp.miplevel - 1 <= level ? 1 : level - floor(level);


	float2 uv = ((float2*)tp.uv)[pidx];
	float2 uv0, uv1;
	int4 idx0 = indexFetch(tp, level0, uv, uv0);
	int4 idx1 = indexFetch(tp, level1, uv, uv1);
	for (int i = 0; i < tp.channel; i++) {
		float out = bilerp(
			tp.texture[level0][idx0.x * tp.channel + i], tp.texture[level0][idx0.y * tp.channel + i],
			tp.texture[level0][idx0.z * tp.channel + i], tp.texture[level0][idx0.w * tp.channel + i], uv0);
		if (flevel > 0) {
			float out1 = bilerp(
				tp.texture[level1][idx1.x * tp.channel + i], tp.texture[level1][idx1.y * tp.channel + i],
				tp.texture[level1][idx1.z * tp.channel + i], tp.texture[level1][idx1.w * tp.channel + i], uv1);
			out = lerp(out, out1, flevel);
		}
		tp.out[pidx * tp.channel + i] = out;
	}
}

void Texturemap::forward(TexturemapParams& tp) {
	cudaMemset(tp.kernel.out, 0, tp.kernel.width * tp.kernel.height * tp.kernel.channel * sizeof(float));
	void* args[] = { &tp.kernel };
	cudaLaunchKernel(TexturemapForwardKernel, tp.grid, tp.block, args, 0, NULL);
}

void Texturemap::init(TexturemapParams& tp, RenderingParams& p, float* dLdout) {
	tp.grad.out = dLdout;
	cudaMalloc(&tp.grad.uv, p.height * p.height * 2 * sizeof(float));
	cudaMalloc(&tp.grad.uvDA, p.height * p.height * 4 * sizeof(float));

	int w = tp.kernel.texwidth, h = tp.kernel.texheight;
	for (int i = 0; i < tp.kernel.miplevel; i++) {
		cudaMalloc(&tp.grad.texture[i], w * h * tp.kernel.channel * sizeof(float));
		w >>= 1; h >>= 1;
	}
}

__device__ __forceinline__ void calculateLevel(const TexturemapKernelParams tp, int pidx, int& level0, int& level1, float& flevel, float4& dleveldda) {
	float4 uvDA = ((float4*)tp.uvDA)[pidx];
	float dsdx = uvDA.x * tp.texwidth;
	float dsdy = uvDA.y * tp.texwidth;
	float dtdx = uvDA.z * tp.texheight;
	float dtdy = uvDA.w * tp.texheight;

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = .5f * (s2 + t2);
	float c2 = b * b - a * a;
	float c = sqrt(c2);


	float level = .5f * log2f(b + c);
	level0 = level <= 0 ? 0 : tp.miplevel - 2 <= level ? tp.miplevel - 2 : (int)floor(level);
	level1 = level <= 1 ? 1 : tp.miplevel - 1 <= level ? tp.miplevel - 1 : (int)floor(level) + 1;
	flevel = level <= 0 ? 0 : tp.miplevel - 1 <= level ? 1 : level - floor(level);

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
			dleveldda = make_float4(0.f, 0.f, 0.f, 0.f);
		}	
	}
}

// s_ = frac(s*width_) => d/ds = d/ds_ * width_
// t_ = frac(t*height_) => d/dt = d/dt_ * height_
// l = frac(level) => dl/dlevel = 1
//
// dL/dX = dL/dc * dc/dX
//
// dc/ds = lerp(lerp(c001-c000, c011-c010, t0) * width0, lerp(c101-c100, c111-c110, t1) * width1, l)
// dc/dt = lerp(lerp(c010-c000, c011-c001, s0) * height0, lerp(c110-c100, c111-c101, s1) * height1, l)
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
__global__ void TexturemapBackwardKernel(const TexturemapKernelParams tp, const TexturemapGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tp.width || py >= tp.height || pz >= tp.depth)return;
	int pidx = px + tp.width * (py + tp.height * pz);
	if (tp.rast[pidx * 4 + 3] < 1.f) {
		((float2*)grad.uv)[pidx] = make_float2(0.f, 0.f);
		((float4*)grad.uvDA)[pidx] = make_float4(0.f, 0.f, 0.f, 0.f);
		return;
	}
	int level0 = 0, level1 = 0;
	float flevel = 0.f;
	float4 dleveldda;
	calculateLevel(tp, pidx, level0, level1, flevel, dleveldda);
	float2 uv = ((float2*)tp.uv)[pidx], uv0, uv1;
	float gu = 0.f, gv = 0.f, gl = 0.f;
	int4 idx0 = indexFetch(tp, level0, uv, uv0);
	int4 idx1 = indexFetch(tp, level1, uv, uv1);

	for (int i = 0; i < tp.channel; i++) {
		float dLdout = grad.out[pidx * tp.channel + i];
		atomicAdd(&grad.texture[level0][idx0.x * tp.channel + i], (1.f - flevel) * (1.f - uv0.x) * (1.f - uv0.y) * dLdout);
		atomicAdd(&grad.texture[level0][idx0.y * tp.channel + i], (1.f - flevel) * uv0.x * (1.f - uv0.y) * dLdout);
		atomicAdd(&grad.texture[level0][idx0.z * tp.channel + i], (1.f - flevel) * (1.f - uv0.x) * uv0.y * dLdout);
		atomicAdd(&grad.texture[level0][idx0.w * tp.channel + i], (1.f - flevel) * uv0.x * uv0.y * dLdout);
		float t00 = tp.texture[level0][idx0.x * tp.channel + i];
		float t01 = tp.texture[level0][idx0.y * tp.channel + i];
		float t10 = tp.texture[level0][idx0.z * tp.channel + i];
		float t11 = tp.texture[level0][idx0.w * tp.channel + i];

		float u = lerp(t01 - t00, t11 - t10, uv0.y) * (tp.texwidth >> level0);
		float v = lerp(t10 - t00, t11 - t01, uv0.x) * (tp.texheight >> level0);
		if (flevel > 0) {
			float l = bilerp(t00, t01, t10, t11, uv0);
			atomicAdd(&grad.texture[level1][idx1.x * tp.channel + i], flevel * (1.f - uv1.x) * (1.f - uv1.y) * dLdout);
			atomicAdd(&grad.texture[level1][idx1.y * tp.channel + i], flevel * uv1.x * (1.f - uv1.y) * dLdout);
			atomicAdd(&grad.texture[level1][idx1.z * tp.channel + i], flevel * (1.f - uv1.x) * uv1.y * dLdout);
			atomicAdd(&grad.texture[level1][idx1.w * tp.channel + i], flevel * uv1.x * uv1.y * dLdout);
			t00 = tp.texture[level1][idx1.x * tp.channel + i];
			t01 = tp.texture[level1][idx1.y * tp.channel + i];
			t10 = tp.texture[level1][idx1.z * tp.channel + i];
			t11 = tp.texture[level1][idx1.w * tp.channel + i];
			u = lerp(u, lerp(t01 - t00, t11 - t10, uv1.y) * (tp.texwidth >> level1), flevel);
			v = lerp(v, lerp(t10 - t00, t11 - t01, uv1.x) * (tp.texheight >> level1), flevel);
			gl += (bilerp(t00, t01, t10, t11, uv1) - l) * dLdout;
		}
		gu += u * dLdout;
		gv += v * dLdout;
	}

	((float2*)grad.uv)[pidx] = make_float2(gu, gv);
	((float4*)grad.uvDA)[pidx] = gl * dleveldda;
}

__global__ void gardAddThrough(const TexturemapKernelParams tp, const TexturemapGradParams grad,int index, int width, int height) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= width || py >= height)return;
	int pidx = px + width * (py + height * pz);
	px <<= 1; py <<= 1;
	width <<= 1; height <<= 1;
	int p00idx = px + width * (py + height * pz);
	int p01idx = p00idx + 1;
	int p10idx = p00idx + width;
	int p11idx = p10idx + 1;

	for (int i = 0; i < tp.channel; i++) {
		float g = grad.texture[index][pidx * tp.channel + i];
		AddNaNcheck(grad.texture[index - 1][p00idx * tp.channel + i], g);
		AddNaNcheck(grad.texture[index - 1][p01idx * tp.channel + i], g);
		AddNaNcheck(grad.texture[index - 1][p10idx * tp.channel + i], g);
		AddNaNcheck(grad.texture[index - 1][p11idx * tp.channel + i], g);
	}
}

void gradSum(TexturemapParams& tp) {
	int w = tp.kernel.texwidth >> tp.kernel.miplevel; int h = tp.kernel.texheight >> tp.kernel.miplevel;
	int i = 0;
	void* args[] = { &tp.kernel, &tp.grad, &i, &w, &h };
	for (i = tp.kernel.miplevel - 1; i > 0; i--) {
		w <<= 1; h <<= 1;
		dim3 block = getBlock(w, h);
		dim3 grid = getGrid(block, w, h);
		cudaLaunchKernel(gardAddThrough, grid, block, args, 0, NULL);
	}
}

void Texturemap::backward(TexturemapParams& tp) {
	int w = tp.kernel.texwidth, h = tp.kernel.texheight;
	for (int i = 0; i < tp.kernel.miplevel; i++) {
		cudaMemset(tp.grad.texture[i], 0, w * h * tp.kernel.channel * sizeof(float));
		w >>= 1; h >>= 1;
	}
	void* args[] = { &tp.kernel, &tp.grad };
	cudaLaunchKernel(TexturemapBackwardKernel, tp.grid, tp.block, args, 0, NULL);
	gradSum(tp);
	buildMipTexture(tp);
}