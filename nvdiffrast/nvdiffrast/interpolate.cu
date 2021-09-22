#include "interpolate.h"

__global__ void interplateFowardKernel(const InterpolateParams ip, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float4 r = ((float4*)ip.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) {
        for (int i = 0; i < ip.dimention; i++) {
            ip.out[pidx * ip.dimention + i] = 0;
            if (ip.enableDA) {
                ((float2*)ip.outDA)[(pidx * ip.dimention + i)] = make_float2(0.0, 0.0);
            }
        }
        return;
    }

    unsigned int idx0 = ip.idx[idx * 3];
    unsigned int idx1 = ip.idx[idx * 3 + 1];
    unsigned int idx2 = ip.idx[idx * 3 + 2];
    for (int i = 0; i < ip.dimention; i++) {
        float a0 = ip.attr[idx0 * ip.dimention + i];
        float a1 = ip.attr[idx1 * ip.dimention + i];
        float a2 = ip.attr[idx2 * ip.dimention + i];
        ip.out[pidx * ip.dimention + i] = a0 * r.x + a1 * r.y + a2 * (1.0 - r.x - r.y);

        if (ip.enableDA) {
            float dadu = a0 - a2;
            float dadv = a1 - a2;
            float4 db = ((float4*)ip.rastDB)[pidx];

            ((float2*)ip.outDA)[(pidx * ip.dimention + i)] =
                make_float2(dadu * db.x + dadv * db.z, dadu * db.y + dadv * db.w);
        }
    }
}

void Interpolate::forwardInit(InterpolateParams& ip, RenderingParams& p, RasterizeParams& rp, Attribute& attr) {
    ip.enableDA = rp.enableDB;
    ip.rast = rp.out;
    ip.attrNum = attr.vboNum;
    ip.idxNum = attr.vaoNum;
    ip.dimention = attr.dimention;
    ip.attr = attr.vbo;
    ip.idx = attr.vao;

    cudaMalloc(&ip.out, p.width * p.height * attr.dimention * sizeof(float));

    if (ip.enableDA) {
        ip.rastDB = rp.outDB;
        cudaMalloc(&ip.outDA, p.width * p.height * attr.dimention * 2 * sizeof(float));
    }
}

void Interpolate::forward(InterpolateParams& ip, RenderingParams& p) {

    interplateFowardKernel << <p.grid, p.block >> > (ip, p);
}

// a = u * (a0 - a2) + v * (a1 - a2) + a2
// dL/da0 = dL/da * da/da0 = dL/da * u
// dL/da1 = dL/da * da/da1 = dL/da * v
// dL/da2 = dL/da * da/da2 = dL/da * (1 - u - v)
//
// dL/du = dot(dL/da, da/du) = dot(dL/da, (a0 - a2))
// dL/dv = dot(dL/da, da/dv) = dot(dL/da, (a1 - a2))
//
//
// da/dx = da/du * du/dx + da/dv * dv/dx = (a0 - a2) * du/dx + (a1 - a2) * dv/dx
// da/dy = da/du * du/dy + da/dv * dv/dy = (a0 - a2) * du/dy + (a1 - a2) * dv/dy
//
// dL/d(du/dx) = dot(dL/d(da/dx), da/du) = dot(dL/d(da/dx), (a0 - a2))
// dL/d(du/dy) = dot(dL/d(da/dy), da/du) = dot(dL/d(da/dy), (a0 - a2))
// dL/d(dv/dx) = dot(dL/d(da/dx), da/dv) = dot(dL/d(da/dx), (a1 - a2))
// dL/d(dv/dy) = dot(dL/d(da/dy), da/dv) = dot(dL/d(da/dy), (a1 - a2))
//
// dL/da0 = dL/d(da/dx) * d(da/dx)/da0 + dL/d(da/dy) * d(da/dy)/da0 = dL/d(da/dx) * du/dx + dL/d(da/dy) * du/dy
// dL/da1 = dL/d(da/dx) * d(da/dx)/da1 + dL/d(da/dy) * d(da/dy)/da1 = dL/d(da/dx) * dv/dx + dL/d(da/dy) * dv/dy
// dL/da2 = dL/d(da/dx) * d(da/dx)/da2 + dL/d(da/dy) * d(da/dy)/da2
//        = -dL/d(da/dx) * du/dx - dL/d(da/dy) * du/dy - dL/d(da/dx) * dv/dx - dL/d(da/dy) * dv/dy = -dL/da0 - dL/da1

__global__ void InterpolateBackwardKernel(const InterpolateParams ip, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float4 r = ((float4*)ip.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) {
        for (int i = 0; i < ip.dimention; i++) {
            ip.gradAttr[pidx * ip.dimention + i] = 0.0;
        }
        ((float2*)ip.gradRast)[pidx] = make_float2(0.0, 0.0);
        return;
    }

    unsigned int idx0 = ip.idx[idx * 3];
    unsigned int idx1 = ip.idx[idx * 3 + 1];
    unsigned int idx2 = ip.idx[idx * 3 + 2];
    float gu = 0;
    float gv = 0;
    for (int i = 0; i < ip.dimention; i++) {
        float dLdout = ip.dLdout[pidx * ip.dimention + i];
        ip.gradAttr[idx0 * ip.dimention + i] = dLdout * r.x;
        ip.gradAttr[idx1 * ip.dimention + i] = dLdout * r.y;
        ip.gradAttr[idx2 * ip.dimention + i] = dLdout * (1.0 - r.x - r.y);
        gu += dLdout * (ip.attr[idx0 * ip.dimention + i] - ip.attr[idx2 * ip.dimention + i]);
        gv += dLdout * (ip.attr[idx1 * ip.dimention + i] - ip.attr[idx2 * ip.dimention + i]);
    }
    ((float2*)ip.gradRast)[pidx] = make_float2(gu, gv);
}

void Interpolate::backwardInit(InterpolateParams& ip, RenderingParams& p, float* dLdout, float* dLdda) {
    ip.dLdout = dLdout;
    ip.dLdda = dLdda;
    cudaMalloc(&ip.gradAttr, ip.attrNum * ip.dimention * sizeof(float));
    cudaMalloc(&ip.gradRast, p.width * p.height * 4 * sizeof(float));
    cudaMalloc(&ip.gradRastDB, p.width * p.height * 4 * sizeof(float));
}

void Interpolate::backwardInit(InterpolateParams& ip, RenderingParams& p, float* dLdout) {
    ip.dLdout = dLdout;
    cudaMalloc(&ip.gradAttr, ip.attrNum * ip.dimention * sizeof(float));
    cudaMalloc(&ip.gradRast, p.width * p.height * 4 * sizeof(float));
}

void Interpolate::backward(InterpolateParams& ip, RenderingParams& p) {
    InterpolateBackwardKernel << <p.grid, p.block>> > (ip, p);
}