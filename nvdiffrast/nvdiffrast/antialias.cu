#include "antialias.h"

__device__ __forceinline__ void forwardEdgeLeak(const AntialiasParams ap, float2 va, float2 vb, float2 pix, float2 opix, int pidx, int oidx, float d) {
    float p = cross(pix - va, pix - vb);
    float o = cross(opix - va, opix - vb);
    if (p * o > 1e-6) return;
    float2 e = va - vb;
    if ((e.x + e.y) * (e.x - e.y) * d > 0)return;
    float alpha = p / (p - o) - 0.5;
    for (int i = 0; i < ap.channel; i++) {
        float diff = ap.in[pidx * ap.channel + i] - ap.in[oidx * ap.channel + i];
        if (alpha > 0) {
            atomicAdd(&ap.out[oidx * ap.channel + i], alpha * diff);
        }
        else{
            atomicAdd(&ap.out[pidx * ap.channel + i], alpha * diff);
        }
    }
}

__device__ __forceinline__ void forwardTriangleFetch(const AntialiasParams ap, const RenderingParams p, int pidx, int oidx, float2 pix, float2 opix, float d) {
    int idx = (int)ap.rast[pidx * 4 + 3] - 1;
    unsigned int idx0 = ap.idx[idx * 3];
    unsigned int idx1 = ap.idx[idx * 3 + 1];
    unsigned int idx2 = ap.idx[idx * 3 + 2];
    float2 p0 = ((float2*)ap.pos)[idx0 * 2];
    float2 p1 = ((float2*)ap.pos)[idx1 * 2];
    float2 p2 = ((float2*)ap.pos)[idx2 * 2];
    float w0 = ap.pos[idx0 * 4 + 3];
    float w1 = ap.pos[idx1 * 4 + 3];
    float w2 = ap.pos[idx2 * 4 + 3];
    p0 *= (1.0 / w0);
    p1 *= (1.0 / w1);
    p2 *= (1.0 / w2);

    forwardEdgeLeak(ap, p0, p1, pix, opix, pidx, oidx, d);
    forwardEdgeLeak(ap, p1, p2, pix, opix, pidx, oidx, d);
    forwardEdgeLeak(ap, p2, p0, pix, opix, pidx, oidx, d);
}

__global__ void antialiasForwardKernel(const AntialiasParams ap, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    for (int i = 0; i < ap.channel; i++) {
        atomicAdd(&ap.out[pidx * ap.channel + i], ap.in[pidx * ap.channel + i]);
    }

    float2 tri0 = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 tri1 = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri0;
    float2 tri2 = py > 0 ? ((float2*)ap.rast)[(pidx - p.width) * 2 + 1] : tri0;

    float2 pix = make_float2(2.0 * (px + 0.5) / p.width - 1.0, 2.0 * (py + 0.5) / p.height - 1.0);
    if (tri1.y != tri0.y) {
        int oidx = pidx - 1;
        float2 opix = make_float2(pix.x - 2.0 / p.width, pix.y);
        if ((int)tri0.y || tri0.x > tri1.x) forwardTriangleFetch(ap, p, pidx, oidx, pix, opix, 1.0);
        else  forwardTriangleFetch(ap, p, oidx, pidx, opix, pix, 1.0);
    }
    if (tri2.y != tri0.y) {
        int oidx = pidx - p.width;
        float2 opix = make_float2(pix.x, pix.y - 2.0 / p.height);
        if ((int)tri0.y || tri0.x > tri2.x) forwardTriangleFetch(ap, p, pidx, oidx, pix, opix, -1.0);
        else  forwardTriangleFetch(ap, p, oidx, pidx, opix, pix, -1.0);
    }
}

void Antialias::forwardInit(AntialiasParams& ap, RenderingParams& p, Attribute& pos, ProjectParams& pp, RasterizeParams& rp, float* in, int channel) {
    ap.channel = channel;
    ap.pos = pp.out;
    ap.idx = pos.vao;
    ap.rast = rp.out;
    ap.in = in;
    ap.posNum = pos.vboNum;
    cudaMalloc(&ap.out, p.width * p.height * channel * sizeof(float));
}

void Antialias::forward(AntialiasParams& ap, RenderingParams& p) {
    cudaMemset(ap.out, 0, p.width * p.height * ap.channel * sizeof(float));
    antialiasForwardKernel << <p.grid, p.block >> > (ap, p);
}

// find edge in pixel pair (p, o, va, vb) and get each pixel color (pin, oin)
// 
// P = cross(p-va, p-vb), O = -cross(o-va, o-vb), D = P - O <=> 1 - P/D = -O/D
// pout = pin + max(O/D - 0.5, 0) * (oin - pin)
// oout = oin + max(0, P/D - 0.5) * (pin - oin)
//
// dL/dpin = dL/dpout * dpout/dpin + dL/doout * doout/dpin
//         = dL/dpout * (1 - max(O/D - 0.5, 0)) + dL/doout * max(0, P/D - 0.5)
// dL/doin = dL/dpout * dpout/doin + dL/doout * doout/doin
//         = dL/dpout * (max(O/D - 0.5, 0)) + dL/doout * (1 - max(0, P/D - 0.5))
//
// dL/dpout,dL/doout either is 0
// dL/d(x,y,w) = dL/dpout * dpout/d(x,y,w) + dL/doout * doout/d(x,y,w)
//
// (F = P/D)' = (F*O' - G*P'))/D
// (G = O/D = -(1-F))' = (F*O' - G*P')/D = F'
//
// dL/dxa = (dL/dpout + dL/doout) * (F*(o-vb).y/wa + G*(p-vb).y/wa)/D
// dL/dya = (dL/dpout + dL/doout) * -(F*(o-vb).x/wa + G*(p-vb).x/wa)/D
// dL/dwa = (dL/dpout + dL/doout) * (F*((o-vb).y*xa - (o-vb).x*ya)/wa^2 + G*((p-vb).y*xa - (p-vb).x*ya)/wa^2)/D
//        = dL/dxa * xa/wa + dL/dya * ya/wa
// dL/dxb = (dL/dpout + dL/doout) * -(F*(o-va).y/wb + G*(p-va).y/wb)/D
// dL/dyb = (dL/dpout + dL/doout) * (F*(o-va).x/wb + G*(p-va).y/wb)/D
// dL/dwb = (dL/dpout + dL/doout) * (F*(-(o-va).y*xb + (o-va).x*yb)/wb^2 + G*(-(p-va).y*xb+(p-va).x*yb)/wa^2)/D
//        = dL/dxb * xb/wb + dL/dyb * yb/wb

__device__ __forceinline__ void backwardEdgeLeak(const AntialiasParams ap, float2 va, float2 vb, float2 pix, float2 opix, int pidx, int oidx, float d) {
    float p = cross(pix - va, pix - vb);
    float o = cross(opix - va, opix - vb);
    if (p * o > 1e-6) return;
    float2 e = va - vb;
    if ((e.x + e.y) * (e.x - e.y) * d > 0)return;

    float alpha = p / (p - o) - 0.5;
    for (int i = 0; i < ap.channel; i++) {
        if (alpha > 0) {
            atomicAdd(&ap.out[oidx * ap.channel + i], alpha * (ap.in[pidx * ap.channel + i] - ap.in[oidx * ap.channel + i]));
        }
        else {
            atomicAdd(&ap.out[pidx * ap.channel + i], alpha * (ap.in[pidx * ap.channel + i] - ap.in[oidx * ap.channel + i]));
        }
    }
}

__device__ __forceinline__ void backwardTriangleFetch(const AntialiasParams ap, const RenderingParams p, int pidx, int oidx, float2 pix, float2 opix, float d) {
    int idx = (int)ap.rast[pidx * 4 + 3] - 1;
    unsigned int idx0 = ap.idx[idx * 3];
    unsigned int idx1 = ap.idx[idx * 3 + 1];
    unsigned int idx2 = ap.idx[idx * 3 + 2];
    float2 p0 = ((float2*)ap.pos)[idx0 * 2];
    float2 p1 = ((float2*)ap.pos)[idx1 * 2];
    float2 p2 = ((float2*)ap.pos)[idx2 * 2];
    float w0 = ap.pos[idx0 * 4 + 3];
    float w1 = ap.pos[idx1 * 4 + 3];
    float w2 = ap.pos[idx2 * 4 + 3];
    p0 *= (1.0 / w0);
    p1 *= (1.0 / w1);
    p2 *= (1.0 / w2);

    backwardEdgeLeak(ap, p0, p1, pix, opix, pidx, oidx, d);
    backwardEdgeLeak(ap, p1, p2, pix, opix, pidx, oidx, d);
    backwardEdgeLeak(ap, p2, p0, pix, opix, pidx, oidx, d);
}
__global__ void antialiasBackwardKernel(const AntialiasParams ap, RenderingParams& p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float2 tri0 = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 tri1 = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri0;
    float2 tri2 = py > 0 ? ((float2*)ap.rast)[(pidx - p.width) * 2 + 1] : tri0;

    float2 pix = make_float2(2.0 * (px + 0.5) / p.width - 1.0, 2.0 * (py + 0.5) / p.height - 1.0);
    if (tri1.y != tri0.y) {
        int oidx = pidx - 1;
        float2 opix = make_float2(pix.x - 2.0 / p.width, pix.y);
        if ((int)tri0.y || tri0.x > tri1.x) backwardTriangleFetch(ap, p, pidx, oidx, pix, opix, 1.0);
        else  backwardTriangleFetch(ap, p, oidx, pidx, opix, pix, 1.0);
    }
    if (tri2.y != tri0.y) {
        int oidx = pidx - p.width;
        float2 opix = make_float2(pix.x, pix.y - 2.0 / p.height);
        if ((int)tri0.y || tri0.x > tri2.x) backwardTriangleFetch(ap, p, pidx, oidx, pix, opix, -1.0);
        else  backwardTriangleFetch(ap, p, oidx, pidx, opix, pix, -1.0);
    }
}

void Antialias::backwardInit(AntialiasParams& ap, RenderingParams& p, float* dLdout) {
    ap.dLdout = dLdout;
    cudaMalloc(&ap.gradPos, ap.posNum * 4 * sizeof(float));
    cudaMalloc(&ap.gradIn, p.width * p.height * ap.channel * sizeof(float));
}

void Antialias::backward(AntialiasParams& ap, RenderingParams& p) {
    cudaMemset(ap.gradPos, 0, ap.posNum * 4 * sizeof(float));
    cudaMemset(ap.gradIn, 0, p.width * p.height * ap.channel * sizeof(float));
    antialiasBackwardKernel << <p.grid, p.block >> > (ap, p);
}