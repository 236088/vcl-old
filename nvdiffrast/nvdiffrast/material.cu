#include "material.h"

void Material::init(MaterialParams& mp, RenderingParams& p, RasterizeParams& rp, InterpolateParams& pos, InterpolateParams& normal, float* in, int channel) {
    mp.channel = channel;
    mp.pos = pos.out;
    mp.normal = normal.out;
    mp.rast = rp.out;
    mp.in = in;
    cudaMalloc(&mp.out, p.width * p.height * channel * sizeof(float));
}
void Material::init(MaterialParams& mp, float* light, int lightNum, float3 eye,  float roughness, float metallic) {
    cudaMalloc(&mp.light, lightNum * 4 * sizeof(float));
    cudaMemcpy(mp.light, light, lightNum * 4 * sizeof(float), cudaMemcpyHostToDevice);
    mp.eye = eye;
    mp.roughness = roughness;
    mp.metallic = metallic;
    mp.lightNum = lightNum;
}

void Material::init(MaterialParams& mp, RenderingParams& p, float* dLdout) {

}

__global__ void MaterialForwardKernel(const MaterialParams mp,const RenderingParams p) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)return;
    int pidx = px + p.width * (py + p.height * pz);

    if (mp.rast[pidx * 4 + 3] < 1.0) return;

    float4 pos= ((float4*)mp.pos)[pidx];
    float4 n = ((float4*)mp.normal)[pidx];
    float4 v = make_float4(mp.eye.x, mp.eye.y, mp.eye.z, 1.) - pos;
    v *= (1. / sqrt(dot(v, v)));
    float nv = clamp(dot(n, v), 1e-3, 1.);
    float a = mp.roughness * mp.roughness;
    float frd = (1. - mp.metallic) / 3.14159265;
    for (int i = 0; i < mp.lightNum; i++) {
        float4 light = ((float4*)mp.light)[i];
        float4 l = light - pos;
        l.w = 0.;
        float l2 = dot(l, l);
        float d = light.w / l2;
        l *= (1. / sqrt(l2));
        float4 h = v + l;
        h *= (1. / sqrt(dot(h, h)));
        float nl = clamp(dot(n, l), 1e-3, 1.);
        float nh = clamp(dot(n, h), 0., 1.);
        float vh = clamp(dot(v, h), 0., 1.);

        float V = 1. / ((nv + sqrt(a + (1. - a) * nv * nv)) * (nl + sqrt(a + (1. - a) * nl * nl)));
        float f = nh * nh * (a - 1.) + 1.;
        V *= a / (3.14159265 * f * f);
        float  pw = pow(1. - vh, 5.);

        for (int i = 0; i < mp.channel; i++) {
            float c = mp.in[pidx * mp.channel + i];
            float frs = c * mp.metallic;
            frs += (1 - frs) * pw;
            mp.out[pidx * mp.channel + i] += nl * d  * ((1. - frs) * frd * c + frs * V);
        }
    }
}

void Material::forward(MaterialParams& mp, RenderingParams& p) {
    cudaMemset(mp.out, 0, p.width * p.height * mp.channel * sizeof(float));
    void* args[] = { &mp, &p };
    cudaLaunchKernel(MaterialForwardKernel, p.grid, p.block, args, 0, NULL);
}

void Material::backward(MaterialParams& mp, RenderingParams& p) {

}