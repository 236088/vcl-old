#include "rasterize.h"

void Rasterize::init(RasterizeParams& rp, RenderingParams& p, float* dLdout, float* dLddb) {
    rp.dLdout = dLdout;
    rp.dLddb = dLddb;
    if (!rp.enableAA)cudaMalloc(&rp.gradPos, rp.posNum * 4 * sizeof(float));
}

void Rasterize::init(RasterizeParams& rp, RenderingParams& p, float* dLdout) {
    rp.dLdout = dLdout;
    if (!rp.enableAA)cudaMalloc(&rp.gradPos, rp.posNum * 4 * sizeof(float));
}

// calculate d[u, v]/d[X, yc, wc]
//
// [p] = [p2 + (p0 - p2) * u + (p1 - p2) * v]
// w = w2 + (w0 - w2) * u + (w1 - w2) * v
//
// [pix] = p / w <=> p = pix * w
//
// [(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)] * [u, v] = [-(w2 * pix - p2)]
// [u, v] = [(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)]^-1 *[-(w2 * pix - p2)]
//
// D = det([(w0 * pix - p0) - (w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2)])
//   = cross(w0 * pix - p0, w1 * pix - p1) + cross(w1 * pix - p1, w2 * pix - p2) + cross(w2 * pix - p2, w0 * pix - p0)
//
// u = cross(-(w2 * pix - p2), (w1 * pix - p1) - (w2 * pix - p2))/D
//   = cross((w1 * pix - p1), (w2 * pix - p2))/D
// v = -cross((w0 * pix - p0), (w2 * pix - p2))/D
//
// (F=f/D)' = f'/D - F * D'/D
// dL/dX = du/dX * dL/du + dv/dX * dL/dv
//       
// dL/dx0 = (w2 * pix - p2).y/D * dL/dv - (u * dL/du + v * dL/dv) * (-(w1 * pix - p1).y + (w2 * pix - p2).y)/D
// dL/dy0 = -(w2 * pix - p2).x/D * dL/dv - (u * dL/du + v * dL/dv) * ((w1 * pix - p1).x - (w2 * pix - p2).x)/D
// dL/dw0 = -dL/dx0 * pix.x - dL/dy0 * pix.y
//
// dL/dx1 = -(w2 * pix - p2).y/D * dL/du - (u * dL/du + v * dL/dv) * ((w0 * pix - p0).y - (w2 * pix - p2).y)/D
// dL/dy1 = (w2 * pix - p2).x/D * dL/du - (u * dL/du + v * dL/dv) * (-(w0 * pix - p0).x + (w2 * pix - p2).x)/D
// dL/dw1 = -dL/dx1 * pix.x - dL/dy1 * pix.y
//
// dL/dx2 = ((w1 * pix - p1).y/D - u * (dD/dx2)/D) * dL/du + (-(w0 * pix - p0).y/D - v * (dD/dx2)/D) * dL/dv
//       = ((w1 * pix - p1).y/D * dL/du - (w0 * pix - p0).y/D * dL/dv) - (u * dL/du + v * dL/dv) * ((w1 * pix - p1).y - (w0 * pix - p0).y)/D
// dL/dy2 = (-(w1 * pix - p1).x/D * dL/du + (w0 * pix - p0).x/D * dL/dv) - (u * dL/du + v * dL/dv) * (-(w1 * pix - p1).x + (w0 * pix - p0).x)/D
// dL/dw2 = -dL/dx2 * pix.x - dL/dy2 * pix.y
//
//
// u = (cross(w1 * p2 - w2 * p1, pix) + cross(p1, p2))/D
// v = (-cross(w0 * p2 - w2 * p0, pix) - cross(p0, p2))/D
//
// D = (w2 - w1) * cross(p0, pix) + (w0 - w2) * cross(p1, pix) + (w1 - w0) * cross(p2, pix) + cross(p0, p1) + cross(p1, p2) + cross(p2, p0)
// dD/dx = (w0 * p1 - w1 * p0).y + (w2 * p0 - w0 * p2).y + (w1 * p2 - w2 * p1).y
// dD/dy = -(w0 * p1 - w1 * p0).x - (w2 * p0 - w0 * p2).x - (w1 * p2 - w2 * p1).x
//
// du/dx = (w1 * p2 - w2 * p1).y/D - cross((w1 * pix - p1), (w2 * pix - p2))/D * ((w2 - w1) * p0.y + (w0 - w2) * p1.y + (w1 - w0) * p2.y)/D
// du/dy = -(w1 * p2 - w2 * p1).x/D - cross((w1 * pix - p1), (w2 * pix - p2))/D * ((w2 - w1) * p0.x + (w0 - w2) * p1.x + (w1 - w0) * p2.x)/D
// dv/dx = -(w0 * p2 - w2 * p0).y/D + cross((w0 * pix - p0), (w2 * pix - p2))/D * ((w2 - w1) * p0.y + (w0 - w2) * p1.y + (w1 - w0) * p2.y)/D
// dv/dy = (w0 * p2 - w2 * p0).x/D + cross((w0 * pix - p0), (w2 * pix - p2))/D * ((w2 - w1) * p0.x + (w0 - w2) * p1.x + (w1 - w0) * p2.x)/D
//
// (f(X) =                  G(X)/D +                              (r(X)=R(X)/D) *                                                    D(X)/D)'
//   =(G(X)/D)' + D(X)/D*(R(X)/D)' + R(X)/D*(D(X)/D)'
//   =(G(X)' - G(X)/D*D')/D + D(X)/D*(R(X)' - R(X)/D*D')/D + R(X)/D*(D(X)'-D(X)/D*D')/D
//   =(G(X)' + D(X)/D*R(X)' + r(X)*D(X)')/D - (2*f(X)-G(X)/D)*D'/D
//
// dL/dX = dL/d(du/dx) * D(du/dx)/dX + dL/d(du/dy) * D(du/dy)/dX + dL/d(dv/dx) * D(dv/dx)/dX + dL/d(dv/dy) * D(dv/dy)/dX
//   = dL/d(du/dx)* ((G(du/dx)' + D(du/dx)/D*R(du/dx)' + r(du/dx)*D(du/dx)')/D - (2*f(du/dx)-G(du/dx)/D)*D'/D)
//   + dL/d(du/dy)* ((G(du/dy)' + D(du/dy)/D*R(du/dy)' + r(du/dy)*D(du/dy)')/D - (2*f(du/dy)-G(du/dy)/D)*D'/D)
//   + dL/d(dv/dx)* ((G(dv/dx)' + D(dv/dx)/D*R(dv/dx)' + r(dv/dx)*D(dv/dx)')/D - (2*f(dv/dx)-G(dv/dx)/D)*D'/D)
//   + dL/d(dv/dy)* ((G(dv/dy)' + D(dv/dy)/D*R(dv/dy)' + r(dv/dy)*D(dv/dy)')/D - (2*f(dv/dy)-G(dv/dy)/D)*D'/D)
//   = -(dL/d(du/dx)*(2*du/dx-G(du/dx)/D) + dL/d(du/dy)*(2*du/dy-G(du/dy)/D) + dL/d(dv/dx)*(2*dv/dx-G(dv/dx)/D) + dL/d(dv/dy)*(2*dv/dy-G(dv/dy)/D))*D'/D
//    + (dL/d(du/dx)*G(du/dx)'/D + dL/d(du/dy)*G(du/dy)'/D + dL/d(dv/dx)*G(dv/dx)'/D + dL/d(dv/dy)*G(dv/dy)'/D)
//    + ((dL/d(du/dx)*(dD/dx)/D + dL/d(du/dy)*(dD/dy)/D) * U'/D + (dL/d(dv/dx)*(dD/dx)/D + dL/d(dv/dy)*(dD/dy)/D) * V'/D)
//    + ((dL/d(du/dx)*u + dL/d(dv/dx)*v) * (dD/dx)'/D + (dL/d(du/dy)*u + dL/d(dv/dy)*v) * (dD/dy)'/D)
//
// dL/dx0 = 
// dL/dy0 = 
// dL/dw0 = 
// dL/dx1 = 
// dL/dy1 = 
// dL/dw1 = 
// dL/dx2 = 
// dL/dy2 = 
// dL/dw2 = 


__global__ void RasterizeBackwardKernel(const RasterizeParams rp, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    float4 r = ((float4*)rp.out)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;
    float2 dLdout = ((float2*)rp.dLdout)[pidx];
    int bitcheck = __float_as_int(dLdout.x) | __float_as_int(dLdout.y);
    if (rp.enableDB) {
        float4 dLddb = ((float4*)rp.dLddb)[pidx];
    }
    if ((bitcheck << 1) == 0)return;

    float2 pix = make_float2(
        2.f * (px + 0.5) / (float)p.width - 1.f,
        2.f * (py + 0.5) / (float)p.height - 1.f
    );
    unsigned int idx0 = rp.idx[idx * 3];
    unsigned int idx1 = rp.idx[idx * 3 + 1];
    unsigned int idx2 = rp.idx[idx * 3 + 2];
    float2 p0 = ((float2*)rp.pos)[idx0 * 2];
    float2 p1 = ((float2*)rp.pos)[idx1 * 2];
    float2 p2 = ((float2*)rp.pos)[idx2 * 2];
    float w0 = rp.pos[idx0 * 4 + 3];
    float w1 = rp.pos[idx1 * 4 + 3];
    float w2 = rp.pos[idx2 * 4 + 3];
    p0 = w0 * pix - p0;
    p1 = w1 * pix - p1;
    p2 = w2 * pix - p2;
    float2 e0 = p0 - p2;
    float2 e1 = p1 - p2;
    float a = cross(e0, e1);

    float eps = a > 0 ? 1e-6f : -1e-6f;
    float ia = 1.f / (a + eps);
    float kuv = r.x * dLdout.x + r.y * dLdout.y;

    float gx0 = (dLdout.y * p2.y + kuv * e1.y) * ia;
    float gy0 = (-dLdout.y * p2.x - kuv * e1.x) * ia;
    float gw0 = -gx0 * pix.x - gy0 * pix.y;
    float gx1 = (-dLdout.x * p2.y - kuv * e0.y) * ia;
    float gy1 = (dLdout.x * p2.x + kuv * e0.x) * ia;
    float gw1 = -gx1 * pix.x - gy1 * pix.y;
    float gx2 = ((-dLdout.y * p0.y + dLdout.x * p1.y) - kuv * (-p0.y + p1.y)) * ia;
    float gy2 = ((dLdout.y * p0.x - dLdout.x * p1.x) - kuv * (p0.x - p1.x)) * ia;
    float gw2 = -gx2 * pix.x - gy2 * pix.y;

    if (rp.enableDB) {

    }

    atomicAdd_xyw(rp.gradPos + idx0 * 4, gx0, gy0, gw0);
    atomicAdd_xyw(rp.gradPos + idx1 * 4, gx1, gy1, gw1);
    atomicAdd_xyw(rp.gradPos + idx2 * 4, gx2, gy2, gw2);
}

void Rasterize::backward(RasterizeParams& rp, RenderingParams& p) {
    if (!rp.enableAA) cudaMemset(rp.gradPos, 0, rp.posNum * 4 * sizeof(float));
    void* args[] = { &rp, &p };
    cudaLaunchKernel(RasterizeBackwardKernel, p.grid, p.block, args, 0, NULL);
}
