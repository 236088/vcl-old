#include "common.h"

void Rendering::init(RenderingParams& rp, int width, int height, int depth) {
    rp.width = width;
    rp.height = height;
    rp.depth = depth;
    rp.block = getBlock(width, height);
    rp.grid = getGrid(rp.block, width, height);
}

void attributeInit(Attribute& attr, float* h_vbo, unsigned int* h_vao, int vboNum, int vaoNum, int dimention) {
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

void loadOBJ(const char* path, Attribute& pos, Attribute& texel, Attribute& normal) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        return;
    }

    std::vector<float> tempPos, tempTexel, tempNorm;
    std::vector<unsigned int> tempPosIndex, tempTexelIndex, tempNormIndex;
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
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            float v[2];
            fscanf(file, "%f %f\n", &v[0], &v[1]);
            tempTexel.push_back(v[0]);
            tempTexel.push_back(v[1]);
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            float v[3];
            fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]);
            tempNorm.push_back(v[0]);
            tempNorm.push_back(v[1]);
            tempNorm.push_back(v[2]);
        }
        else if (strcmp(lineHeader, "f") == 0) {
            unsigned int idx[9];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &idx[0], &idx[1], &idx[2], &idx[3], &idx[4], &idx[5], &idx[6], &idx[7], &idx[8]);
            if (matches != 9) {
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return;
            }
            tempPosIndex.push_back(idx[0] - 1);
            tempPosIndex.push_back(idx[3] - 1);
            tempPosIndex.push_back(idx[6] - 1);
            tempTexelIndex.push_back(idx[1] - 1);
            tempTexelIndex.push_back(idx[4] - 1);
            tempTexelIndex.push_back(idx[7] - 1);
            tempNormIndex.push_back(idx[2] - 1);
            tempNormIndex.push_back(idx[5] - 1);
            tempNormIndex.push_back(idx[8] - 1);
        }
    }

    if (&pos != nullptr)  attributeInit(pos, tempPos.data(), tempPosIndex.data(), tempPos.size() / 3, tempPosIndex.size() / 3, 3);
    if (&texel != nullptr)  attributeInit(texel, tempTexel.data(), tempTexelIndex.data(), tempTexel.size() / 2, tempTexelIndex.size() / 3, 2);
    if (&normal != nullptr)  attributeInit(normal, tempNorm.data(), tempNormIndex.data(), tempNorm.size() / 3, tempNormIndex.size() / 3, 3);
}

dim3 getBlock(int width, int height) {
    if (width * height > MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK) {
        return dim3(MAX_DIM_PER_BLOCK, MAX_DIM_PER_BLOCK);
    }
    else {
        dim3 block;
        while (block.x * block.y < width * height) {
            block.x <<= 1;
            block.y <<= 1;
        }
        return block;
    }
}

dim3 getGrid(dim3 block, int width, int height) {
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    return grid;
}