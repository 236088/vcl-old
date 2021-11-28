#include "buffer.h"

void Attribute::init(Attribute& attr, float* h_vbo, unsigned int* h_vao, int vboNum, int vaoNum, int dimention) {
    attr.dimention = dimention;
    attr.vboNum = vboNum;
    attr.vaoNum = vaoNum;
    cudaMallocHost(&attr.h_vbo, vboNum * dimention * sizeof(float));
    cudaMallocHost(&attr.h_vao, vaoNum * 3 * sizeof(unsigned int));
    cudaMalloc(&attr.vbo, vboNum * dimention * sizeof(float));
    cudaMalloc(&attr.vao, vaoNum * 3 * sizeof(float));
    cudaMemcpy(attr.h_vbo, h_vbo, vboNum * dimention * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(attr.h_vao, h_vao, vaoNum * 3 * sizeof(float), cudaMemcpyHostToHost);
    cudaMemcpy(attr.vbo, h_vbo, vboNum * dimention * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(attr.vao, h_vao, vaoNum * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void Attribute::loadOBJ(const char* path, Attribute& pos, Attribute& texel, Attribute& normal) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        return;
    }

    std::vector<float> tempPos, tempTexel, tempNorm;
    std::vector<unsigned int> tempPosIndex, tempTexelIndex, tempNormIndex;
    int posNum = 0, texelNum = 0, normNum = 0, indexNum = 0;
    while (1) {
        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break;
        if (strcmp(lineHeader, "v") == 0) {
            float v[3];
            fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]);
            tempPos.push_back(v[0]);
            tempPos.push_back(v[1]);
            tempPos.push_back(v[2]);
            posNum++;
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            float v[2];
            fscanf(file, "%f %f\n", &v[0], &v[1]);
            tempTexel.push_back(v[0]);
            tempTexel.push_back(v[1]);
            texelNum++;
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            float v[3];
            fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]);
            tempNorm.push_back(v[0]);
            tempNorm.push_back(v[1]);
            tempNorm.push_back(v[2]);
            normNum++;
        }
        else if (strcmp(lineHeader, "f") == 0 && posNum > 0) {
            unsigned int idx[9];
            if (texelNum > 0 && normNum > 0) {
                int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &idx[0], &idx[3], &idx[6], &idx[1], &idx[4], &idx[7], &idx[2], &idx[5], &idx[8]);
                if (matches != 9) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempTexelIndex.push_back(idx[3] - 1);
                tempTexelIndex.push_back(idx[4] - 1);
                tempTexelIndex.push_back(idx[5] - 1);
                tempNormIndex.push_back(idx[6] - 1);
                tempNormIndex.push_back(idx[7] - 1);
                tempNormIndex.push_back(idx[8] - 1);
            }
            else if (texelNum > 0) {
                int matches = fscanf(file, "%d/%d %d/%d %d/%d\n", &idx[0], &idx[3], &idx[1], &idx[4], &idx[2], &idx[5]);
                if (matches != 6) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempTexelIndex.push_back(idx[3] - 1);
                tempTexelIndex.push_back(idx[4] - 1);
                tempTexelIndex.push_back(idx[5] - 1);
            }
            else if (normNum > 0) {
                int matches = fscanf(file, "%d//%d %d//%d %d//%d\n", &idx[0], &idx[6], &idx[1], &idx[7], &idx[2], &idx[8]);
                if (matches != 6) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempNormIndex.push_back(idx[6] - 1);
                tempNormIndex.push_back(idx[7] - 1);
                tempNormIndex.push_back(idx[8] - 1);
            }
            else {
                int matches = fscanf(file, "%d %d %d\n", &idx[0], &idx[1], &idx[2]);
                if (matches != 3) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
            }
            tempPosIndex.push_back(idx[0] - 1);
            tempPosIndex.push_back(idx[1] - 1);
            tempPosIndex.push_back(idx[2] - 1);
            indexNum++;
        }
    }


    if (posNum > 0)  Attribute::init(pos, tempPos.data(), tempPosIndex.data(), posNum, indexNum, 3);
    if (texelNum > 0)  Attribute::init(texel, tempTexel.data(), tempTexelIndex.data(), texelNum, indexNum, 2);
    if (normNum > 0)  Attribute::init(normal, tempNorm.data(), tempNormIndex.data(), normNum, indexNum, 3);
}

__global__ void Shrink(const Attribute attr, float* buf, float s) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= attr.vaoNum)return;
    int idx0 = attr.vao[px * 3];
    int idx1 = attr.vao[px * 3 + 1];
    int idx2 = attr.vao[px * 3 + 2];
    for (int i = 0; i < attr.dimention; i++) {
        atomicAdd(&attr.vbo[idx0 * attr.dimention + i], (buf[idx1 * attr.dimention + i]- buf[idx0 * attr.dimention + i]) * s);
        atomicAdd(&attr.vbo[idx1 * attr.dimention + i], (buf[idx2 * attr.dimention + i]- buf[idx1 * attr.dimention + i]) * s);
        atomicAdd(&attr.vbo[idx2 * attr.dimention + i], (buf[idx0 * attr.dimention + i]- buf[idx2 * attr.dimention + i]) * s);
    }
}

void Attribute::posShrink(Attribute& pos, float s, int repeat) {
    dim3 block = dim3(pos.vaoNum > 1024 ? 1024 : pos.vaoNum);
    dim3 grid = dim3((pos.vaoNum - 1) / 1024 + 1);
    float* buf;
    cudaMalloc(&buf, pos.vboNum * pos.dimention * sizeof(float));
    void* args[] = { &pos,&buf, &s};
    for (int i = 0; i < repeat; i++) {
        cudaMemcpy(buf, pos.vbo, pos.vboNum * pos.dimention * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaLaunchKernel(Shrink, block, grid, args, 0, NULL);
    }
    cudaFree(buf);
}