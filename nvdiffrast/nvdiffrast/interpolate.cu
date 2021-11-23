#include "interpolate.h"

void commonInit(InterpolateParams& p, RenderBuffer& intr, RenderBuffer& rast, Attribute& attr) {
    p.params.width = intr.width;
    p.params.height = intr.height;
    p.params.depth = intr.depth;
    p.params.rast = rast.buffer;
    p.params.attr = attr.vbo;
    p.params.idx = attr.vao;
    p.params.dimention = attr.dimention;
    p.params.out = intr.buffer;
    p.block = getBlock(intr.width, intr.height);
    p.grid = getGrid(p.block, intr.width, intr.height, intr.depth);
}

void Interpolate::init(InterpolateParams& p, RenderBuffer& intr, RenderBuffer& rast, Attribute& attr) {
    commonInit(p, intr, rast, attr);
    p.params.enableDA = false;
}

void Interpolate::init(InterpolateParams& p, RenderBuffer& intr, RenderBuffer& intrDA, RenderBuffer& rast, RenderBuffer& rastDB, Attribute& attr) {
    commonInit(p, intr, rast, attr);
    p.params.enableDA = true;
    p.params.rastDB = rastDB.buffer;
    p.params.outDA = intrDA.buffer;
}

__global__ void InterplateForwardKernel(const InterpolateKernelParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float4 r = ((float4*)p.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    unsigned int idx0 = p.idx[idx * 3];
    unsigned int idx1 = p.idx[idx * 3 + 1];
    unsigned int idx2 = p.idx[idx * 3 + 2];
    for (int i = 0; i < p.dimention; i++) {
        float a0 = p.attr[idx0 * p.dimention + i];
        float a1 = p.attr[idx1 * p.dimention + i];
        float a2 = p.attr[idx2 * p.dimention + i];
        p.out[pidx * p.dimention + i] = a0 * r.x + a1 * r.y + a2 * (1.0 - r.x - r.y);

        if (p.enableDA) {
            float dadu = a0 - a2;
            float dadv = a1 - a2;
            float4 db = ((float4*)p.rastDB)[pidx];

            ((float2*)p.outDA)[(pidx * p.dimention + i)] =
                make_float2(dadu * db.x + dadv * db.z, dadu * db.y + dadv * db.w);
        }
    }
}

void Interpolate::forward(InterpolateParams& p) {
    void* args[] = { &p.params };
    cudaLaunchKernel(InterplateForwardKernel, p.grid, p.block, args, 0, NULL);
}

void commonInit(InterpolateGradParams& p, RenderBufferGrad& intr, RenderBufferGrad& rast, AttributeGrad& attr) {
    p.grad.out = intr.grad;
    p.grad.rast = rast.grad;
    p.grad.attr = attr.grad;
}

void Interpolate::init(InterpolateGradParams& p, RenderBufferGrad& intr, RenderBufferGrad& rast, AttributeGrad& attr) {
    init((InterpolateParams&)p, intr, rast, attr);
    commonInit(p, intr, rast, attr);
}

void Interpolate::init(InterpolateGradParams& p, RenderBufferGrad& intr, RenderBufferGrad& intrDA, RenderBufferGrad& rast, RenderBufferGrad& rastDB, AttributeGrad& attr) {
    init((InterpolateParams&)p, intr, intrDA, rast, rastDB, attr);
    commonInit(p, intr, rast, attr);
    p.grad.outDA = intrDA.grad;
    p.grad.rastDB = rastDB.grad;
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

__global__ void InterpolateBackwardKernel(const InterpolateKernelParams p,const InterpolateKernelGradParams g) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float4 r = ((float4*)p.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) {
        ((float2*)g.rast)[pidx] = make_float2(0.0, 0.0);
        return;
    }

    unsigned int idx0 = p.idx[idx * 3];
    unsigned int idx1 = p.idx[idx * 3 + 1];
    unsigned int idx2 = p.idx[idx * 3 + 2];
    float gu = 0;
    float gv = 0;
    for (int i = 0; i < p.dimention; i++) {
        float dLdout = g.out[pidx * p.dimention + i];
        atomicAdd(&g.attr[idx0 * p.dimention + i], dLdout * r.x);
        atomicAdd(&g.attr[idx1 * p.dimention + i], dLdout * r.y);
        atomicAdd(&g.attr[idx2 * p.dimention + i], dLdout * (1.0 - r.x - r.y));
        gu += dLdout * (p.attr[idx0 * p.dimention + i] - p.attr[idx2 * p.dimention + i]);
        gv += dLdout * (p.attr[idx1 * p.dimention + i] - p.attr[idx2 * p.dimention + i]);
    }
    ((float2*)g.rast)[pidx] = make_float2(gu, gv);
}

void Interpolate::backward(InterpolateGradParams& p) {
    void* args[] = { &p.params, &p.grad };
    cudaLaunchKernel(InterpolateBackwardKernel, p.grid, p.block, args, 0, NULL);
}