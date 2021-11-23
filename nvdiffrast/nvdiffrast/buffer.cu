#include "buffer.h"

void Attribute ::init(Attribute& attr, int vboNum, int dimention, int vaoNum) {
    attr.dimention = dimention;
    attr.vboNum = vboNum;
    attr.vaoNum = vaoNum;
    cudaMalloc(&attr.vbo, attr.vboSize());
    cudaMalloc(&attr.vao, attr.vaoSize());
}

void Attribute ::clear(Attribute& attr) {
    cudaMemset(attr.vbo, 0, attr.vboSize());
}

void Attribute::loadOBJ(const char* path, Attribute* pos, Attribute* texel, Attribute* normal) {
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

    if (pos != nullptr && posNum > 0) {
        Attribute::init(*pos, posNum, 3, indexNum);
        cudaMemcpy((*pos).vbo, tempPos.data(), (*pos).vboSize(), cudaMemcpyHostToDevice);
        cudaMemcpy((*pos).vao, tempPosIndex.data(), (*pos).vaoSize(), cudaMemcpyHostToDevice);
    }
    if (texel != nullptr && texelNum > 0) {
        Attribute::init(*texel, texelNum, 2, indexNum);
        cudaMemcpy((*texel).vbo, tempPos.data(), (*texel).vboSize(), cudaMemcpyHostToDevice);
        cudaMemcpy((*texel).vao, tempPosIndex.data(), (*texel).vaoSize(), cudaMemcpyHostToDevice);
    }
    if (normal != nullptr && normNum > 0) {
        Attribute::init(*normal, normNum, 3, indexNum);
        cudaMemcpy((*normal).vbo, tempPos.data(), (*normal).vboSize(), cudaMemcpyHostToDevice);
        cudaMemcpy((*normal).vao, tempPosIndex.data(), (*normal).vaoSize(), cudaMemcpyHostToDevice);
    }
}

void AttributeGrad ::init(AttributeGrad& attr, int vboNum, int dimention, int vaoNum) {
    Attribute::init(attr, vboNum, dimention, vaoNum);
    cudaMalloc(&attr.grad, attr.vboSize());
}

void AttributeGrad::clear(AttributeGrad& attr, bool alsoBuffer) {
    cudaMemset(attr.grad, 0, attr.vboSize());
    if (alsoBuffer)Attribute::clear(attr);
}

void AttributeHost::init(AttributeHost& attr, Attribute& src) {
    attr.dimention = src.dimention;
    attr.vboNum = src.vboNum;
    attr.vaoNum = src.vaoNum;
    cudaMallocHost(&attr.vbo, attr.vboSize());
    cudaMallocHost(&attr.vao, attr.vaoSize());
}

void AttributeHost::vboMemcpyIn(AttributeHost& attr, Attribute& src) {
    cudaMemcpy(attr.vbo, src.vbo, attr.vboSize(), cudaMemcpyDeviceToHost);
}

void AttributeHost::vaoMemcpyIn(AttributeHost& attr, Attribute& src) {
    cudaMemcpy(attr.vao, src.vao, attr.vaoSize(), cudaMemcpyDeviceToHost);
}

void AttributeHost::vboMemcpyOut(AttributeHost& attr, Attribute& dst) {
    cudaMemcpy(dst.vbo, attr.vbo, attr.vboSize(), cudaMemcpyHostToDevice);
}

void AttributeHost::vaoMemcpyOut(AttributeHost& attr, Attribute& dst) {
    cudaMemcpy(dst.vao, attr.vao, attr.vaoSize(), cudaMemcpyHostToDevice);
}



void RenderBuffer::init(RenderBuffer& buf, int width, int height, int channel, int depth) {
    buf.width = width;
    buf.height = height;
    buf.channel = channel;
    buf.depth = depth;
    cudaMalloc(&buf.buffer, buf.Size());
}

void RenderBuffer::clear(RenderBuffer& buf) {
    cudaMemset(buf.buffer, 0, buf.Size());
}

void RenderBufferGrad::init(RenderBufferGrad& buf, int width, int height, int channel, int depth) {
    RenderBuffer::init(buf, width, height, channel, depth);
    cudaMalloc(&buf.grad, buf.Size());
}

void RenderBufferGrad::clear(RenderBufferGrad& buf, bool alsoBuffer) {
    cudaMemset(buf.grad, 0, buf.Size());
    if (alsoBuffer)RenderBuffer::clear(buf);
}

void RenderBufferHost::init(RenderBufferHost& buf, RenderBuffer& src) {
    buf.width = src.width;
    buf.height = src.height;
    buf.channel = src.channel;
    buf.depth = src.depth;
    cudaMallocHost(&buf.buffer, buf.Size());
}

void RenderBufferHost::MemcpyIn(RenderBufferHost& buf, RenderBuffer& src) {
    cudaMemcpy(buf.buffer, src.buffer, buf.Size(), cudaMemcpyDeviceToHost);
}

void RenderBufferHost::MemcpyOut(RenderBufferHost& buf, RenderBuffer& dst) {
    cudaMemcpy(dst.buffer, buf.buffer, buf.Size(), cudaMemcpyHostToDevice);
}



void MipTexture::init(MipTexture& miptex, int width, int height, int channel, int miplevel) {
    miplevel = miplevel < TEX_MAX_MIP_LEVEL ? miplevel : TEX_MAX_MIP_LEVEL;
    if (((width >> miplevel) << miplevel) != width || ((height >> miplevel) << miplevel) != height) {
        printf("Invalid miplevel value");
        exit(1);
    }
    miptex.width = width;
    miptex.height = height;
    miptex.channel = channel;
    miptex.miplevel = miplevel;
    for (int i = 0; i < miplevel; i++) {
        cudaMalloc(&miptex.texture[i], miptex.Size(i));
    }
}

void MipTexture::clear(MipTexture& miptex) {
    for (int i = 0; i < miptex.miplevel; i++) {
        cudaMemset(&miptex.texture[i], 0, miptex.Size(i));
    }
}

__global__ void ucharToFloat(unsigned char* data, const MipTexture miptex) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= miptex.width || py >= miptex.height || pz >= miptex.channel)return;
    int pidx = pz + miptex.channel * (px + miptex.width * py);
    miptex.texture[0][pidx] = (float)data[pidx] / 255.0;
}

__global__ void downSampling(const MipTexture miptex, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= miptex.width || py >= miptex.height || pz >= miptex.channel)return;
    int pidx = pz + miptex.channel * (px + miptex.width * py);
    int p00idx = (px << 1) + (width << 1) * (py << 1);
    int p01idx = p00idx + 1;
    int p10idx = p00idx + (width << 1);
    int p11idx = p10idx + 1;
    
    int i = index - 1;
    float p00 = miptex.texture[i][p00idx * miptex.channel + pz];
    float p01 = miptex.texture[i][p01idx * miptex.channel + pz];
    float p10 = miptex.texture[i][p10idx * miptex.channel + pz];
    float p11 = miptex.texture[i][p11idx * miptex.channel + pz];

    float p = (p00 + p01 + p10 + p11) * 0.25;
    miptex.texture[index][pidx] = p;
}

void MipTexture::buildMIP(MipTexture& miptex) {
    int w = miptex.width, h = miptex.height;
    int i = 0;
    void* args[] = { &miptex, &i, &w, &h };
    for (i = 1; i < miptex.miplevel; i++) {
        w >>= 1; h >>= 1;
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h, miptex.channel);
        cudaError_t e=cudaLaunchKernel(downSampling, grid, block, args, 0, NULL);
        printf(" %s %p\n", cudaGetErrorString(e), args);
    }
}

void MipTexture::loadBMP(const char* path, MipTexture& miptex, int miplevel) {
    unsigned char header[54];

    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("Image could not be opened\n");
        return;
    }
    if (fread(header, 1, 54, file) != 54) {
        printf("Not a correct BMP file\n");
        return;
    }
    if (header[0] != 'B' || header[1] != 'M') {
        printf("Not a correct BMP file\n");
        return;
    }
    unsigned int dataPos = *(int*)&(header[0x0A]);
    unsigned int imageSize = *(int*)&(header[0x22]);
    unsigned int width = *(int*)&(header[0x12]);
    unsigned int height = *(int*)&(header[0x16]);
    unsigned int channel = *(int*)&(header[0x1c]) / sizeof(unsigned char);
    MipTexture::init(miptex, width, height, channel, miplevel);

    if (imageSize == 0)    imageSize = width * height * channel;
    if (dataPos == 0)      dataPos = 54;
    fseek(file, dataPos, SEEK_SET);

    unsigned char* data = new unsigned char[imageSize];
    fread(data, 1, imageSize, file);
    fclose(file);

    unsigned char* dev_data;

    cudaMalloc(&dev_data, imageSize * sizeof(unsigned char));
    cudaMemcpy(dev_data, data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    void* args[] = { &dev_data,&miptex };
    dim3 block = getBlock(width, height);
    dim3 grid = getGrid(block, width, height, channel);
    cudaLaunchKernel(ucharToFloat, grid, block, args, 0, NULL);
    cudaFree(dev_data);
    printf(" %s\n", cudaGetErrorString(cudaGetLastError()));

    MipTexture::buildMIP(miptex);
    printf(" %s\n", cudaGetErrorString(cudaGetLastError()));
}

void MipTextureGrad::init(MipTextureGrad& miptex, int width, int height, int channel, int miplevel) {
    MipTexture::init(miptex, width, height, channel, miplevel);
}

void MipTextureGrad::clear(MipTextureGrad& miptex, bool alsoBuffer) {
    for (int i = 0; i < miptex.miplevel; i++) {
        cudaMemset(&miptex.grad[i], 0, miptex.Size(i));
    }
    if (alsoBuffer)MipTexture::clear(miptex);
}

__global__ void gardAddThrough(const MipTextureGrad miptex, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= miptex.width || py >= miptex.height || pz >= miptex.channel)return;
    int pidx = pz + miptex.channel * (px + miptex.width * py);
    int p00idx = (px << 1) + (width << 1) * (py << 1);
    int p01idx = p00idx + 1;
    int p10idx = p00idx + (width << 1);
    int p11idx = p10idx + 1;

    float g = miptex.grad[index][pidx];
    --index;
    if (!isnan(g)) {
        AddNaNcheck(miptex.grad[index][p00idx * miptex.channel + pz], g);
        AddNaNcheck(miptex.grad[index][p01idx * miptex.channel + pz], g);
        AddNaNcheck(miptex.grad[index][p10idx * miptex.channel + pz], g);
        AddNaNcheck(miptex.grad[index][p11idx * miptex.channel + pz], g);
    }
}

void MipTextureGrad::gradSum(MipTextureGrad& miptex) {
    int w = miptex.width >> miptex.miplevel; int h = miptex.height >> miptex.miplevel;
    int i = 0;
    void* args[] = { &miptex, &i, &w, &h };
    for (i = miptex.miplevel - 1; i > 0; i--) {
        w <<= 1; h <<= 1;
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h, miptex.channel);
        cudaLaunchKernel(gardAddThrough, grid, block, args, 0, NULL);
    }
}