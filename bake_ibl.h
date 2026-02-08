#pragma once

#include <vector>

struct TextureData
{
    int width;
    int height;
    int mipCount;
    int numFaces;             // 1 for 2D, 6 for Cubemap
    std::vector<float> data;  // RGBA32F, layout: mip -> face -> y -> x -> rgba
};

void BakeIrradianceCUDA(const TextureData& src, TextureData& dst, int sampleCount);
void BakeRadianceCUDA(const TextureData& src, TextureData& dst, int sampleCount);
