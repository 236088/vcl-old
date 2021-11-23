#include "common.h"

dim3 getBlock(int width, int height) {
    if (width  >= MAX_DIM_PER_BLOCK && height >= MAX_DIM_PER_BLOCK) {
        return dim3(MAX_DIM_PER_BLOCK, MAX_DIM_PER_BLOCK);
    }
    if (width * height > MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK) {
        if (width < height) {
            return dim3(width, (MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK) / width);
        }
        else {
            return dim3((MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK) / height, height);
        }
    }
    else {
        return dim3(width, height);
    }
}

dim3 getBlock(int width, int height, int channel) {
    int maxWidth = MAX_DIM_PER_BLOCK;
    int maxHeight = MAX_DIM_PER_BLOCK;
    for (int i = 0; (1 << i) < channel; i++) {
        if (i & 1)maxWidth >>= 1;
        else maxHeight >>= 1;
    }
    if (width  >= maxWidth && height >= maxHeight) {
        return dim3(maxWidth, maxHeight, channel);
    }
    if (width * height > maxWidth * maxHeight) {
        if (width < height) {
            return dim3(width, (maxWidth * maxHeight) / width, channel);
        }
        else {
            return dim3((maxWidth * maxHeight) / height, height, channel);
        }
    }
    else {
        return dim3(width, height);
    }
}

dim3 getGrid(dim3 block, int width, int height, int depth) {
    return dim3(
        (width - 1) / block.x + 1,
        (height - 1) / block.y + 1,
        depth
    );
}

void cudaErrorCheck(const char* id, cudaError_t status) {
    printf("%s: %d, %s\n",id, (int)status, cudaGetErrorString(status));
}