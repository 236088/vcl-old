#include "material.h"

void Material::init(MaterialParams& mp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in) {
    mp.kernel.width = p.width;
    mp.kernel.height = p.height;
    mp.kernel.depth = p.depth;
    mp.kernel.pos = pos.kernel.out;
    mp.kernel.normal = normal.kernel.out;
    mp.kernel.rast = rp.kernel.out;
    mp.kernel.in = in;
    cudaMalloc(&mp.kernel.out, p.width * p.height * 3 * sizeof(float));
    mp.block = getBlock(p.width, p.height);
    mp.grid = getGrid(mp.block, p.width, p.height);
}

void Material::init(MaterialParams& mp, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess) {
    mp.kernel.eye = eye;
    mp.kernel.lightNum = lightNum;
    cudaMalloc(&mp.kernel.lightpos, lightNum * sizeof(float3));
    cudaMemcpy(mp.kernel.lightpos, lightpos, lightNum * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&mp.kernel.lightintensity, (lightNum + 1) * sizeof(float3));
    cudaMemcpy(mp.kernel.lightintensity, &ambient, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(&mp.kernel.lightintensity[1], lightintensity, lightNum * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&mp.kernel.params, 4 * sizeof(float));
    float params[4]{ Ka, Kd,  Ks,  shininess };
    cudaMemcpy(mp.kernel.params, params, 4 * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void MaterialForwardKernel(const MaterialKernelParams mp) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mp.width || py >= mp.height || pz >= mp.depth)return;
    int pidx = px + mp.width * (py + mp.height * pz);

    if (mp.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = ((float3*)mp.pos)[pidx];
    float3 n = ((float3*)mp.normal)[pidx];
    float3 v = *(float3*)&mp.eye - pos;
    v *= (1.f / sqrt(dot(v, v)));
    float3 diffuse = make_float3(0.f, 0.f, 0.f);
    float3 specular = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < mp.lightNum; i++) {
        float3 lightpos = mp.lightpos[i];
        float3 l = lightpos - pos;
        l *= (1.f / sqrt(dot(l, l)));
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float3 intensity = mp.lightintensity[i + 1];
        diffuse += intensity * max(ln, 0.f);
        float powrv = pow(max(rv, 0.f), mp.params[3]);
        AddNaNcheck(specular.x, intensity.x * powrv);
        AddNaNcheck(specular.y, intensity.y * powrv);
        AddNaNcheck(specular.z, intensity.z * powrv);
    }
    float Ka = mp.params[0];
    float Kd = mp.params[1];
    float Ks = mp.params[2];
    ((float3*)mp.out)[pidx] = ((float3*)mp.in)[pidx] * (Ka * mp.lightintensity[0] + Kd * diffuse + Ks * specular);
}

void Material::forward(MaterialParams& mp) {
    cudaMemset(mp.kernel.out, 0, mp.kernel.width * mp.kernel.height * 3 * sizeof(float));
    void* args[] = { &mp.kernel};
    cudaLaunchKernel(MaterialForwardKernel, mp.grid, mp.block, args, 0, NULL);
}

void Material::init(MaterialParams& mp, RenderingParams& p, float* dLdout) {
    mp.grad.out = dLdout;
    cudaMalloc(&mp.grad.in, p.width * p.height * 3 * sizeof(float));
    cudaMalloc(&mp.grad.lightpos, mp.kernel.lightNum * sizeof(float3));
    cudaMalloc(&mp.grad.lightintensity, (mp.kernel.lightNum + 1) * sizeof(float3));
    cudaMalloc(&mp.grad.params, 4 * sizeof(float));
}

__global__ void MaterialBackwardKernel(const MaterialKernelParams mp, const MaterialGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mp.width || py >= mp.height || pz >= mp.depth)return;
    int pidx = px + mp.width * (py + mp.height * pz);

    if (mp.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = ((float3*)mp.pos)[pidx];
    float3 n = ((float3*)mp.normal)[pidx];
    float3 v = *(float3*)&mp.eye - pos;
    v *= (1. / sqrt(dot(v, v)));
    float3 diffuse = make_float3(0.f, 0.f, 0.f);
    float3 specular = make_float3(0.f, 0.f, 0.f);
    float dshine = 0.f;
    for (int i = 0; i < mp.lightNum; i++) {
        float3 light = mp.lightpos[i];
        //dl/dlight=1
        //dln/dl_=-n_*l_/dot(l,l)/sqrt(dot(l, l))
        float3 l = light - pos;
        l *= (1.f / sqrt(dot(l, l)));
        float ln = dot(l, n);
        //dr/dl_=2*n_*n-1
        //drv/dr=v
        //dspec/drv=shininess*pow(rv,shininess-1)
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float3 intensity = mp.lightintensity[i + 1];
        diffuse += intensity * max(ln, 0.f);
        float powrv = pow(max(rv, 0.f), mp.params[3]);
        AddNaNcheck(specular.x, intensity.x * powrv);
        AddNaNcheck(specular.y, intensity.y * powrv);
        AddNaNcheck(specular.z, intensity.z * powrv);
        AddNaNcheck(dshine, (intensity.x + intensity.y + intensity.z) * log(max(rv, 0.f)) * powrv);
    }
    float Ka = mp.params[0];
    float Kd = mp.params[1];
    float Ks = mp.params[2];
    float3 dLdout = ((float3*)grad.out)[pidx];
    float3 din = dLdout * ((float3*)mp.in)[pidx];
    atomicAdd(&grad.params[0], dot(mp.lightintensity[0], din));
    atomicAdd(&grad.params[1], dot(din,diffuse));
    atomicAdd(&grad.params[2], dot(din,specular));
    atomicAdd(&grad.params[3], Ks * dshine);
    ((float3*)grad.in)[pidx] = dLdout * (Ka * mp.lightintensity[0] + Kd * diffuse + Ks * specular);
}

void Material::backward(MaterialParams& mp) {
    cudaMemset(mp.grad.in, 0, mp.kernel.width * mp.kernel.height * 3 * sizeof(float));
    cudaMemset(&mp.grad.lightpos, 0, mp.kernel.lightNum * sizeof(float3));
    cudaMemset(&mp.grad.lightintensity, 0, (mp.kernel.lightNum + 1) * sizeof(float3));
    cudaMemset(&mp.grad.params, 0, 4 * sizeof(float));
    void* args[] = { &mp.kernel, &mp.grad };
    cudaLaunchKernel(MaterialBackwardKernel, mp.grid, mp.block, args, 0, NULL);
}