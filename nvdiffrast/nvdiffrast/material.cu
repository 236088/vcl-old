#include "material.h"

void Material::init(MaterialParams& mp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in) {
    mp.pos = pos.out;
    mp.normal = normal.out;
    mp.rast = rp.out;
    mp.in = in;
    cudaMalloc(&mp.out, p.width * p.height * 3 * sizeof(float));
}

void Material::init(MaterialParams& mp, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess) {
    mp.eye = eye;
    mp.lightNum = lightNum;
    cudaMalloc(&mp.lightpos, lightNum * sizeof(float3));
    cudaMemcpy(mp.lightpos, lightpos, lightNum * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&mp.lightintensity, (lightNum + 1) * sizeof(float3));
    cudaMemcpy(mp.lightintensity, &ambient, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(&mp.lightintensity[1], lightintensity, lightNum * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&mp.params, 4 * sizeof(float));
    float params[4]{ Ka, Kd,  Ks,  shininess };
    cudaMemcpy(mp.params, params, 4 * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void MaterialForwardKernel(const MaterialParams mp, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    if (mp.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = *(float3*)&mp.pos[pidx * 4];
    float3 n = *(float3*)&mp.normal[pidx * 4];
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

void Material::forward(MaterialParams& mp, RenderingParams& p) {
    cudaMemset(mp.out, 0, p.width * p.height * 3 * sizeof(float));
    void* args[] = { &mp, &p };
    cudaLaunchKernel(MaterialForwardKernel, p.grid, p.block, args, 0, NULL);
}

void Material::init(MaterialParams& mp, RenderingParams& p, float* dLdout) {
    mp.dLdout = dLdout;
    cudaMalloc(&mp.gradIn, p.width * p.height * 3 * sizeof(float));
    cudaMalloc(&mp.gradLightpos, mp.lightNum * sizeof(float3));
    cudaMalloc(&mp.gradLightintensity, (mp.lightNum + 1) * sizeof(float3));
    cudaMalloc(&mp.gradParams, 4 * sizeof(float));
}

__global__ void MaterialBackwardKernel(const MaterialParams mp, const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    if (mp.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = *(float3*)&mp.pos[pidx * 4];
    float3 n = *(float3*)&mp.normal[pidx * 4];
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
    float3 dLdout = ((float3*)mp.dLdout)[pidx];
    float3 din = dLdout * ((float3*)mp.in)[pidx];
    atomicAdd(&mp.gradParams[0], dot(mp.lightintensity[0], din));
    atomicAdd(&mp.gradParams[1], dot(din,diffuse));
    atomicAdd(&mp.gradParams[2], dot(din,specular));
    atomicAdd(&mp.gradParams[3], Ks * dshine);
    ((float3*)mp.gradIn)[pidx] = dLdout * (Ka * mp.lightintensity[0] + Kd * diffuse + Ks * specular);
}

void Material::backward(MaterialParams& mp, RenderingParams& p) {
    cudaMemset(mp.gradIn, 0, p.width * p.height * 3 * sizeof(float));
    cudaMemset(&mp.gradLightpos, 0, mp.lightNum * sizeof(float3));
    cudaMemset(&mp.gradLightintensity, 0, (mp.lightNum + 1) * sizeof(float3));
    cudaMemset(&mp.gradParams, 0, 4 * sizeof(float));
    void* args[] = { &mp, &p };
    cudaLaunchKernel(MaterialBackwardKernel, p.grid, p.block, args, 0, NULL);
}