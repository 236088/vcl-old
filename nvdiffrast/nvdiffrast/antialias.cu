#include "antialias.h"

void Antialias::init(AntialiasParams& ap, ProjectParams& pp, RasterizeParams& rp, float* in, int channel) {
    ap.kernel.width = rp.kernel.width;
    ap.kernel.height = rp.kernel.height;
    ap.kernel.depth = rp.kernel.depth;
    ap.kernel.channel = channel;
    ap.kernel.proj = pp.kernel.out;
    ap.kernel.idx = pp.vao;
    ap.kernel.rast = rp.kernel.out;
    ap.kernel.in = in;
    ap.projNum = pp.kernel.vboNum;
    ap.kernel.xh = rp.kernel.width / 2.f;
    ap.kernel.yh = rp.kernel.height / 2.f;
    CUDA_ERROR_CHECK(cudaMalloc(&ap.kernel.out, rp.kernel.width * rp.kernel.height * channel * sizeof(float)));

    ap.block = rp.block;
    ap.grid = rp.grid;
    if (rp.kernel.width > rp.kernel.height) {
        ap.block.x >>= 1;
        ap.grid.x <<= 1;
    }
    else {
        ap.block.y >>= 1;
        ap.grid.y <<= 1;
    }
}

__device__ __forceinline__ void forwardEdgeLeak(const AntialiasKernelParams ap, int pidx, int oidx, float2 pa, float2 pb, float2 o) {
    float a = cross(pa, pb);
    float oa = cross(pa - o, pb - o);
    if (a * oa > 0) return;
    float2 e = pa - pb;
    float n = (o.x + o.y) * (o.x - o.y);
    if ((e.x + e.y) * (e.x - e.y) * n > 0)return;
    e *= (o.x + o.y);
    float D = n > 0 ? -e.y : e.x;
    float ia =1.f / (D + (D > 0 ? 1e-3 : -1e-3));
    float alpha = a * ia - .5f;
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

__device__ __forceinline__ void forwardTriangleFetch(const AntialiasKernelParams ap, int pidx, int oidx, float2 f, int d) {
    int idx = (int)ap.rast[pidx * 4 + 3] - 1;
    unsigned int idx0 = ap.idx[idx * 3];
    unsigned int idx1 = ap.idx[idx * 3 + 1];
    unsigned int idx2 = ap.idx[idx * 3 + 2];
    float2 v0 = ((float2*)ap.proj)[idx0 * 2];
    float2 v1 = ((float2*)ap.proj)[idx1 * 2];
    float2 v2 = ((float2*)ap.proj)[idx2 * 2];
    float iw0 =1.f / ap.proj[idx0 * 4 + 3];
    float iw1 =1.f / ap.proj[idx1 * 4 + 3];
    float iw2 =1.f / ap.proj[idx2 * 4 + 3];
    float2 o = make_float2(d - 1, -d);
    if (pidx < oidx) {
        f += o;
        o = -o;
    }
    float2 p0, p1, p2;
    p0.x = (v0.x * iw0 +1.f) * ap.xh - f.x;
    p0.y = (v0.y * iw0 +1.f) * ap.yh - f.y;
    p1.x = (v1.x * iw1 +1.f) * ap.xh - f.x;
    p1.y = (v1.y * iw1 +1.f) * ap.yh - f.y;
    p2.x = (v2.x * iw2 +1.f) * ap.xh - f.x;
    p2.y = (v2.y * iw2 +1.f) * ap.yh - f.y;
    forwardEdgeLeak(ap, pidx, oidx, p0, p1, o);
    forwardEdgeLeak(ap, pidx, oidx, p1, p2, o);
    forwardEdgeLeak(ap, pidx, oidx, p2, p0, o);
}

__global__ void AntialiasForwardKernel(const AntialiasKernelParams ap) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= ap.width || py >= ap.height || pz >= ap.depth)return;
    int pidx = px + ap.width * (py + ap.height * pz);
    for (int i = 0; i < ap.channel; i++) {
        ap.out[pidx * ap.channel + i] = ap.in[pidx * ap.channel + i];
    }
    float2 tri = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)ap.rast)[(pidx - ap.width) * 2 + 1] : tri;
    float2 f = make_float2((float)px + .5f, (float)py + .5f);

    if (trih.y != tri.y) {
        int oidx = pidx - 1;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)trih.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > trih.x)opidx = oidx;
            else ppidx = oidx;
        }
        forwardTriangleFetch(ap, ppidx, opidx, f, 0);
    }
    if (triv.y != tri.y) {
        int oidx = pidx - ap.width;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)triv.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > triv.x)opidx = oidx;
            else ppidx = oidx;
        }
        forwardTriangleFetch(ap, ppidx, opidx, f, 1);
    }
}

void Antialias::forward(AntialiasParams& ap) {
    CUDA_ERROR_CHECK(cudaMemset(ap.kernel.out, 0, ap.kernel.width * ap.kernel.height * ap.kernel.channel * sizeof(float)));
    void* args[] = { &ap.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(AntialiasForwardKernel, ap.grid, ap.block, args, 0, NULL));
}

void Antialias::init(AntialiasParams& ap, RasterizeParams& rp, float* dLdout) {
    ap.grad.out = dLdout;
    CUDA_ERROR_CHECK(cudaMalloc(&ap.grad.proj, ap.projNum * 4 * sizeof(float)));
    rp.grad.proj = ap.grad.proj;
    rp.enableAA = 1;
    CUDA_ERROR_CHECK(cudaMalloc(&ap.grad.in, rp.kernel.width * rp.kernel.height * ap.kernel.channel * sizeof(float)));
}

// horizontal :0, vertical 1;
// 
// (pa, pb) crossed edge (va, vb) screen scale and (fx,fy) to orgin
// pa.x = (va.x/va.w+1.f)*width/2 - (0.5+px)
// pa.y = (va.y/va.w+1.f)*height/2 - (0.5+py)
// pb.x = (vb.x/vb.w+1.f)*width/2 - (0.5+px)
// pb.y = (vb.y/vb.w+1.f)*height/2 - (0.5+py)
// 
// if horizontal (1,0)
//   D = pb.y - pa.y
//   d0 = (pa.x*pb.y-pa.y*pb.x) /D
//   d1 = -((pa.x-1)*pb.y-pa.y*(pb.x-1)) /D
//      =1-d0
// if vertical (0,1)
//   D = pa.x - pb.x
//   d0 = (pa.x*pb.y-pa.y*pb.x) /D
//   d1 = -(pa.x*(pb.y-1)-(pa.y-1)*pb.x) /D
//      =1-d0
// D = dot(pa - pb, o);
//
//  
// if d0 - 0.5 > 0 then
//     pout = pin
//     oout = oin + (d0 - 0.5) * (pin - oin)
//     dL/dout = dL/doout
// else then
//     pout = pin + (d0 - 0.5) * (pin - oin)
//     oout = oin
//     dL/dout = dL/dpout
//
// dL/dpin = dL/dpout * dpout/dpin + dL/doout * doout/dpin
//         = dL/dpout + dL/dout * (d0 - 0.5)
// dL/doin = dL/dpout * dpout/doin + dL/doout * doout/doin
//         = dL/doout - dL/dout * (d0 - 0.5)
//
// (f = F/D)' = (F' - f*D')/D
// 
// dL/d(x,y,w) = dL/dpout * dpout/d(x,y,w) + dL/doout * doout/d(x,y,w)
//         =  dL/dout * (pin - oin) * dd0/d(x,y,w) 
// 
// dpa.x/dva.x=width/2/va.w
// dpa.y/dva.y=height/2/va.w
// dpa.x/dva.w=-width/2*va.x/va.w^2=dpa.x/dva.x*pa.x
// dpa.y/dva.w=-height/2*va.y/va.w^2=dpa.y/dva.y*pa.y
// dpb.x/dvb.x=width/2/vb.w
// dpb.y/dvb.y=height/2/vb.w
// dpb.x/dvb.w=-width/2*vb.x/vb.w^2=dpb.x/dvb.x*pb.x
// dpb.y/dvb.w=-height/2*ba.y/vb.w^2=dpb.y/dvb.y*pb.y
// 
// if horizontal
//   dL/dva.x=dL/dout * (pin - oin)/(pb.y - pa.y) * width/2/va.w * pb.y
//   dL/dva.y=dL/dout * (pin - oin)/(pb.y - pa.y) * height/2/va.w * pb.y * (pa.x - pb.x)/(pb.y - pa.y)
//   dL/dvb.x=dL/dout * (pin - oin)/(pb.y - pa.y) * width/2/vb.w * -pa.y
//   dL/dvb.y=dL/dout * (pin - oin)/(pb.y - pa.y) * height/2/vb.w * -pa.y * (pa.x - pb.x)/(pb.y - pa.y)
//   dL/dva.w=-dL/dva.x*pa.x-dL/dva.y*pa.y
//   dL/dvb.w=-dL/dvb.x*pb.x-dL/dvb.y*pb.y
// if vertical
//   dL/dva.x=dL/dout * (pin - oin)/(pa.x - pb.x) * width/2/va.w * -pb.x * (pb.y - pa.y)/(pa.x - pb.x)
//   dL/dva.y=dL/dout * (pin - oin)/(pa.x - pb.x) * height/2/va.w * -pb.x
//   dL/dvb.x=dL/dout * (pin - oin)/(pa.x - pb.x) * width/2/vb.w * pa.x * (pb.y - pa.y)/(pa.x - pb.x)
//   dL/dvb.y=dL/dout * (pin - oin)/(pa.x - pb.x) * height/2/vb.w * pa.x
//   dL/dva.w=-dL/dva.x*pa.x-dL/dva.y*pa.y
//   dL/dvb.w=-dL/dvb.x*pb.x-dL/dvb.y*pb.y
//

__device__ __forceinline__ void backwardEdgeLeak(const AntialiasKernelParams ap, const AntialiasKernelGradParams grad, int pidx, int oidx, float2 pa, float2 pb, int idxa, int idxb, float iwa, float iwb, float2 o) {
    float a = cross(pa, pb);
    float oa = cross(pa - o, pb - o);
    if (a * oa > 0) return;
    float2 e = pa - pb;
    float n = (o.x + o.y) * (o.x - o.y);
    if ((e.x + e.y) * (e.x - e.y) * n > 0)return;
    e *= (o.x + o.y);
    float D = n > 0 ? -e.y : e.x;
    float ia =1.f / (D + (D > 0 ? 1e-3 : -1e-3));
    float alpha = a * ia - .5f;
    float d = 0.f;
    for (int i = 0; i < ap.channel; i++) {
        float dLdout = alpha > 0 ? grad.out[oidx * ap.channel + i] : grad.out[pidx * ap.channel + i];
        float diff = ap.in[pidx * ap.channel + i] - ap.in[oidx * ap.channel + i];
        atomicAdd(&grad.in[pidx * ap.channel + i], alpha * dLdout);
        atomicAdd(&grad.in[oidx * ap.channel + i], -alpha * dLdout);
        d += dLdout * diff;
    }
    d *= ia;
    float dLdax = d * ap.xh * iwa;
    float dLday = d * ap.yh * iwa;
    float dLdbx = d * ap.xh * iwb;
    float dLdby = d * ap.yh * iwb;

    float r = n > 0 ? e.x : -e.y;
    r *= ia;
    if (n > 0) {
        dLdax *= pb.y;
        dLday *= pb.y * r;
        dLdbx *= -pa.y;
        dLdby *= -pa.y * r;
    }
    else {
        dLdax *= -pb.x * r;
        dLday *= -pb.x;
        dLdbx *= pa.x * r;
        dLdby *= pa.x;

    }
    atomicAdd_xyw(grad.proj + idxa * 4, dLdax, dLday, -dLdax * pa.x - dLday * pa.y);
    atomicAdd_xyw(grad.proj + idxb * 4, dLdbx, dLdby, -dLdbx * pb.x - dLdby * pb.y);
}

__device__ __forceinline__ void backwardTriangleFetch(const AntialiasKernelParams ap, const AntialiasKernelGradParams grad, int pidx, int oidx, float2 f, int d) {
    int idx = (int)ap.rast[pidx * 4 + 3] - 1;
    unsigned int idx0 = ap.idx[idx * 3];
    unsigned int idx1 = ap.idx[idx * 3 + 1];
    unsigned int idx2 = ap.idx[idx * 3 + 2];
    float2 v0 = ((float2*)ap.proj)[idx0 * 2];
    float2 v1 = ((float2*)ap.proj)[idx1 * 2];
    float2 v2 = ((float2*)ap.proj)[idx2 * 2];
    float iw0 =1.f / ap.proj[idx0 * 4 + 3];
    float iw1 =1.f / ap.proj[idx1 * 4 + 3];
    float iw2 =1.f / ap.proj[idx2 * 4 + 3];
    float2 o = make_float2(d - 1, -d);
    if (pidx < oidx) {
        f += o;
        o = -o;
    }
    float2 p0, p1, p2;
    p0.x = (v0.x * iw0 +1.f) * ap.xh - f.x;
    p0.y = (v0.y * iw0 +1.f) * ap.yh - f.y;
    p1.x = (v1.x * iw1 +1.f) * ap.xh - f.x;
    p1.y = (v1.y * iw1 +1.f) * ap.yh - f.y;
    p2.x = (v2.x * iw2 +1.f) * ap.xh - f.x;
    p2.y = (v2.y * iw2 +1.f) * ap.yh - f.y;

    backwardEdgeLeak(ap, grad, pidx, oidx, p0, p1, idx0, idx1, iw0, iw1, o);
    backwardEdgeLeak(ap, grad, pidx, oidx, p1, p2, idx1, idx2, iw1, iw2, o);
    backwardEdgeLeak(ap, grad, pidx, oidx, p2, p0, idx2, idx0, iw2, iw0, o);
}

__global__ void AntialiasBackwardKernel(const AntialiasKernelParams ap, const AntialiasKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= ap.width || py >= ap.height || pz >= ap.depth)return;
    int pidx = px + ap.width * (py + ap.height * pz);
    for (int i = 0; i < ap.channel; i++) {
        grad.in[pidx * ap.channel + i] = grad.out[pidx * ap.channel + i];
    }

    float2 tri = ((float2*)ap.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)ap.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)ap.rast)[(pidx - ap.width) * 2 + 1] : tri;
    float2 f = make_float2((float)px + .5f, (float)py + .5f);

    if (trih.y != tri.y) {
        int oidx = pidx - 1;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)tri.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > trih.x)opidx = oidx;
            else ppidx = oidx;
        }
        backwardTriangleFetch(ap, grad, ppidx, opidx, f, 0);
    }
    if (triv.y != tri.y) {
        int oidx = pidx - ap.width;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)triv.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > triv.x)opidx = oidx;
            else ppidx = oidx;
        }
        backwardTriangleFetch(ap, grad, ppidx, opidx, f, 1);
    }
}

void Antialias::backward(AntialiasParams& ap) {
    CUDA_ERROR_CHECK(cudaMemset(ap.grad.proj, 0, ap.projNum * 4 * sizeof(float)));
    void* args[] = { &ap.kernel,&ap.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel( AntialiasBackwardKernel, ap.grid, ap.block , args, 0, NULL));
}