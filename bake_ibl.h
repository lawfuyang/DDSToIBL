#pragma once

#include <vector>

struct Cubemap
{
    int size;                 // face width == height
    int mipCount;
    std::vector<float> data;  // RGBA32F, layout: mip -> face -> y -> x -> rgba
};

void BakeIrradianceCUDA(const Cubemap& src, Cubemap& dst, int numFaces);
void BakeRadianceCUDA(const Cubemap& src, Cubemap& dst, int numFaces);
