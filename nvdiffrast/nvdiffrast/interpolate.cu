#include "interpolate.h"

void Interpolate::init(InterpolateParams& ip, RasterizeParams& rp, Attribute& attr) {
    ip.kernel.width = rp.kernel.width;
    ip.kernel.height = rp.kernel.height;
    ip.kernel.depth = rp.kernel.depth;
    ip.kernel.enableDA = rp.kernel.enableDB;
    ip.kernel.rast = rp.kernel.out;
    ip.attrNum = attr.vboNum;
    ip.idxNum = attr.vaoNum;
    ip.kernel.dimention = attr.dimention;
    ip.kernel.attr = attr.vbo;
    ip.kernel.idx = attr.vao;

    CUDA_ERROR_CHECK(cudaMalloc(&ip.kernel.out, rp.kernel.width * rp.kernel.height * attr.dimention * sizeof(float)));
    if (ip.kernel.enableDA) {
        ip.kernel.rastDB = rp.kernel.outDB;
        CUDA_ERROR_CHECK(cudaMalloc(&ip.kernel.outDA, rp.kernel.width * rp.kernel.height * attr.dimention * 2 * sizeof(float)));
    }
    ip.block = getBlock(rp.kernel.width, rp.kernel.height);
    ip.grid = getGrid(ip.block, rp.kernel.width, rp.kernel.height);
}

void Interpolate::init(InterpolateParams& ip, RasterizeParams& rp, Attribute& attr, ProjectParams& pp) {
    ip.kernel.width = rp.kernel.width;
    ip.kernel.height = rp.kernel.height;
    ip.kernel.depth = rp.kernel.depth;
    ip.kernel.enableDA = rp.kernel.enableDB;
    ip.kernel.rast = rp.kernel.out;
    ip.kernel.attr = pp.kernel.out;
    ip.attrNum = pp.kernel.size;
    ip.kernel.dimention = pp.kernel.dimention;
    ip.kernel.idx = attr.vao;
    ip.idxNum = attr.vaoNum;

    CUDA_ERROR_CHECK(cudaMalloc(&ip.kernel.out, rp.kernel.width * rp.kernel.height * pp.kernel.dimention * sizeof(float)));
    if (ip.kernel.enableDA) {
        ip.kernel.rastDB = rp.kernel.outDB;
        CUDA_ERROR_CHECK(cudaMalloc(&ip.kernel.outDA, rp.kernel.width * rp.kernel.height * pp.kernel.dimention * 2 * sizeof(float)));
    }
    ip.block = rp.block;
    ip.grid = rp.grid;
}

__global__ void InterplateForwardKernel(const InterpolateKernelParams ip) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= ip.width || py >= ip.height || pz >= ip.depth)return;
    int pidx = px + ip.width * (py + ip.height * pz);

    float4 r = ((float4*)ip.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

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

            ((float2*)ip.outDA)[pidx * ip.dimention + i] =
                make_float2(dadu * db.x + dadv * db.z, dadu * db.y + dadv * db.w);
        }
    }
}

void Interpolate::forward(InterpolateParams& ip) {
    CUDA_ERROR_CHECK(cudaMemset(ip.kernel.out, 0, ip.kernel.width * ip.kernel.height * ip.kernel.dimention * sizeof(float)));
    if (ip.kernel.enableDA) {
        CUDA_ERROR_CHECK(cudaMemset(ip.kernel.outDA, 0, ip.kernel.width * ip.kernel.height * ip.kernel.dimention * 2 * sizeof(float)));
    }
    void* args[] = { &ip.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(InterplateForwardKernel, ip.grid, ip.block, args, 0, NULL));
}

void Interpolate::init(InterpolateParams& ip, Attribute& attr, float* dLdout) {
    ip.grad.out = dLdout;
    CUDA_ERROR_CHECK(cudaMalloc(&ip.grad.attr, ip.attrNum * ip.kernel.dimention * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&ip.grad.rast, ip.kernel.width * ip.kernel.height * 4 * sizeof(float)));
}

void Interpolate::init(InterpolateParams& ip, Attribute& attr, float* dLdout, float* dLdda) {
    init(ip, attr, dLdout);
    ip.grad.outDA = dLdda;
    CUDA_ERROR_CHECK(cudaMalloc(&ip.grad.rastDB, ip.kernel.width * ip.kernel.height * 4 * sizeof(float)));
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

__global__ void InterpolateBackwardKernel(const InterpolateKernelParams ip, const InterpolateKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= ip.width || py >= ip.height || pz >= ip.depth)return;
    int pidx = px + ip.width * (py + ip.height * pz);

    float4 r = ((float4*)ip.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) {
        ((float2*)grad.rast)[pidx] = make_float2(0.0, 0.0);
        return;
    }

    unsigned int idx0 = ip.idx[idx * 3];
    unsigned int idx1 = ip.idx[idx * 3 + 1];
    unsigned int idx2 = ip.idx[idx * 3 + 2];
    float gu = 0;
    float gv = 0;
    for (int i = 0; i < ip.dimention; i++) {
        float dLdout = grad.out[pidx * ip.dimention + i];
        atomicAdd(&grad.attr[idx0 * ip.dimention + i], dLdout * r.x);
        atomicAdd(&grad.attr[idx1 * ip.dimention + i], dLdout * r.y);
        atomicAdd(&grad.attr[idx2 * ip.dimention + i], dLdout * (1.0 - r.x - r.y));
        gu += dLdout * (ip.attr[idx0 * ip.dimention + i] - ip.attr[idx2 * ip.dimention + i]);
        gv += dLdout * (ip.attr[idx1 * ip.dimention + i] - ip.attr[idx2 * ip.dimention + i]);
    }
    ((float2*)grad.rast)[pidx] = make_float2(gu, gv);
}

void Interpolate::backward(InterpolateParams& ip) {
    CUDA_ERROR_CHECK(cudaMemset(ip.grad.attr, 0, ip.attrNum * ip.kernel.dimention * sizeof(float)));
    void* args[] = { &ip.kernel,&ip.grad};
    CUDA_ERROR_CHECK(cudaLaunchKernel(InterpolateBackwardKernel, ip.grid, ip.block, args, 0, NULL));
}