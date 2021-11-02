#include "antialias.h"

__device__ __forceinline__ void forwardEdgeLeak(const AntialiasParams ap, float2 va, float2 vb, float2 pix, float2 opix, int pidx, int oidx, float d) {
    float p = cross(pix - va, pix - vb);
    float o = cross(opix - va, opix - vb);
    if (p * o > 0) return;
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
    float2 v0 = p0 * (1.0 / w0);
    float2 v1 = p1 * (1.0 / w1);
    float2 v2 = p2 * (1.0 / w2);

    forwardEdgeLeak(ap, v0, v1, pix, opix, pidx, oidx, d);
    forwardEdgeLeak(ap, v1, v2, pix, opix, pidx, oidx, d);
    forwardEdgeLeak(ap, v2, v0, pix, opix, pidx, oidx, d);
}

__global__ void antialiasForwardKernel(const AntialiasParams ap, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);
    float2 pix = make_float2(2.0 * (px + 0.5) / p.width - 1.0, 2.0 * (py + 0.5) / p.height - 1.0);

    float2 tri = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)ap.rast)[(pidx - p.width) * 2 + 1] : tri;


    if (trih.y != tri.y) {
        int ppidx = pidx;
        int opidx = pidx;
        float2 ppix = pix;
        float2 opix = pix;
        if ((int)tri.y || tri.x > trih.x) {
            opidx -= 1;
            opix.x -= 2.0 / p.width;
        }
        else {
            ppidx -= 1;
            ppix.x -= 2.0 / p.width;
        }
        forwardTriangleFetch(ap, p, ppidx, opidx, ppix, opix, 1.0);
    }
    if (triv.y != tri.y) {
        int ppidx = pidx;
        int opidx = pidx;
        float2 ppix = pix;
        float2 opix = pix;
        if ((int)tri.y || tri.x > triv.x) {
            opidx -= p.width;
            opix.y -= 2.0 / p.height;
        }
        else {
            ppidx -= p.width;
            ppix.y -= 2.0 / p.height;
        }
        forwardTriangleFetch(ap, p, ppidx, opidx, ppix, opix, -1.0);
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
    cudaMemcpy(ap.out, ap.in, p.width * p.height * ap.channel * sizeof(float), cudaMemcpyDeviceToDevice);
    antialiasForwardKernel << <p.grid, p.block >> > (ap, p);
}

// find edge in pixel pair (p, o, va, vb) and get each pixel color (pin, oin)
// 
// P = cross(p-va, p-vb), O = -cross(o-va, o-vb), D = P - O <=> 1 - P/D = -O/D
// if P/D - 0.5 > 0 then
//     pout = pin
//     oout = oin + (P/D - 0.5) * (pin - oin)
//     dL/dout = dL/doout
// else then
//     pout = pin + (P/D - 0.5) * (pin - oin)
//     oout = oin
//     dL/dout = dL/dpout
//
// dL/dpin = dL/dpout * dpout/dpin + dL/doout * doout/dpin
//         = dL/dpout + dL/dout * (P/D - 0.5)
// dL/doin = dL/dpout * dpout/doin + dL/doout * doout/doin
//         = dL/doout - dL/dout * (P/D - 0.5)
//
// (F = P/D)' = (P' - F*D')/D = (P' - F*(P' - O'))/D = (F*O' + G*P')/D
// (G = 1-F = -O/D)' = -F'
// 
// dL/d(x,y,w) = dL/dpout * dpout/d(x,y,w) + dL/doout * doout/d(x,y,w)
// 
// dL/dxa = dL/dout * (F*(o-vb).y/wa - G*(p-vb).y)/wa)/D * (pin - oin)
// dL/dya = dL/dout * (-F*(o-vb).x/wa + G*(p-vb).x/wa)/D * (pin - oin)
// dL/dwa = dL/dout * (F*(-(o-vb).y*xa/wa^2 + (o-vb).x*ya/wa^2) + G*((p-vb).y*xa/wa^2 - (p-vb).x*ya/wa^2))/D * (pin - oin)
//        = -dL/dxa * va.x - dL/dya * va.y
// dL/dxb = dL/dout * (-F*(o-va).y/wb + G*(p-va).y/wb)/D * (pin - oin)
// dL/dyb = dL/dout * (F*(o-va).x/wb - G*(p-va).x/wb)/D * (pin - oin)
// dL/dwb = dL/dout * (F*((o-va).y*xb/wb^2 - (o-va).x*yb/wb^2) + G*(-(p-va).y*xb/wb^2 + (p-va).x*yb/wb^2))/D * (pin - oin)
//        = -dL/dxb * vb.x - dL/dyb * vb.y
//

__device__ __forceinline__ void backwardEdgeLeak(const AntialiasParams ap, float2 va, float2 vb, float wa, float wb, unsigned int idxa, unsigned int idxb, float2 pix, float2 opix, int pidx, int oidx, float d) {
    float p = cross(pix - va, pix - vb);
    float o = cross(opix - va, opix - vb);
    if (p * o > 0) return;
    float2 e = va - vb;
    if ((e.x + e.y) * (e.x - e.y) * d > 0)return;

    float alpha = p / (p - o) - 0.5;
    float _alpha = 1.0 - alpha;
    float D = p - o;
    for (int i = 0; i < ap.channel; i++) {
        float dLdout = alpha > 0 ? ap.dLdout[oidx * ap.channel + i] : ap.dLdout[pidx * ap.channel + i];
        atomicAdd(&ap.gradIn[pidx * ap.channel + i], dLdout * alpha);
        atomicAdd(&ap.gradIn[oidx * ap.channel + i], -dLdout * alpha);

        float k = dLdout / D * (ap.in[pidx * ap.channel + i] - ap.in[oidx * ap.channel + i]);

        float dLdxa = (alpha * (opix.y - vb.y) - _alpha * (pix.y - vb.y)) * k / wa;
        float dLdya = (-alpha * (opix.x - vb.x) + _alpha * (pix.x - vb.x)) * k / wa;
        float dLdxb = (-alpha * (opix.y - va.y) + _alpha * (pix.y - va.y)) * k / wb;
        float dLdyb = (alpha * (opix.x - va.x) - _alpha * (pix.x - va.x)) * k / wb;

        atomicAdd_xyw(ap.gradPos + idxa * 4, dLdxa, dLdya, -dLdxa * va.x - dLdya * va.y);
        atomicAdd_xyw(ap.gradPos + idxb * 4, dLdxb, dLdyb, -dLdxb * vb.x - dLdyb * vb.y);
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
    float2 v0 = p0 * (1.0 / w0);
    float2 v1 = p1 * (1.0 / w1);
    float2 v2 = p2 * (1.0 / w2);

    backwardEdgeLeak(ap, v0, v1, w0, w1, idx0, idx1, pix, opix, pidx, oidx, d);
    backwardEdgeLeak(ap, v1, v2, w1, w2, idx1, idx2, pix, opix, pidx, oidx, d);
    backwardEdgeLeak(ap, v2, v0, w2, w0, idx2, idx0, pix, opix, pidx, oidx, d);
}

__global__ void antialiasBackwardKernel(const AntialiasParams ap, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float2 tri = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)ap.rast)[(pidx - p.width) * 2 + 1] : tri;

    float2 pix = make_float2(2.0 * (px + 0.5) / p.width - 1.0, 2.0 * (py + 0.5) / p.height - 1.0);
    if (trih.y != tri.y) {
        int ppidx = pidx;
        int opidx = pidx;
        float2 ppix = pix;
        float2 opix = pix;
        if ((int)tri.y || tri.x > trih.x) {
            opidx -= 1;
            opix.x -= 2.0 / p.width;
        }
        else {
            ppidx -= 1;
            ppix.x -= 2.0 / p.width;
        }
        backwardTriangleFetch(ap, p, ppidx, opidx, ppix, opix, 1.0);
    }
    if (triv.y != tri.y) {
        int ppidx = pidx;
        int opidx = pidx;
        float2 ppix = pix;
        float2 opix = pix;
        if ((int)tri.y || tri.x > triv.x) {
            opidx -= p.width;
            opix.y -= 2.0 / p.height;
        }
        else {
            ppidx -= p.width;
            ppix.y -= 2.0 / p.height;
        }
        backwardTriangleFetch(ap, p, ppidx, opidx, ppix, opix, -1.0);
    }
}

void Antialias::backwardInit(AntialiasParams& ap, RenderingParams& p, RasterizeParams& rp, float* dLdout) {
    ap.dLdout = dLdout;
    cudaMalloc(&ap.gradPos, ap.posNum * 4 * sizeof(float));
    rp.gradPos = ap.gradPos;
    rp.enableAA = 1;
    cudaMalloc(&ap.gradIn, p.width * p.height * ap.channel * sizeof(float));
}

void Antialias::backward(AntialiasParams& ap, RenderingParams& p) {
    cudaMemset(ap.gradPos, 0, ap.posNum * 4 * sizeof(float));
    cudaMemcpy(ap.gradIn, ap.dLdout, p.width * p.height * ap.channel * sizeof(float), cudaMemcpyDeviceToDevice);
    antialiasBackwardKernel << <p.grid, p.block >> > (ap, p);
}