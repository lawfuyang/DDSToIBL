#include "bake_ibl.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>

// Constants
const int COMPONENTS_PER_PIXEL_CUDA = 4;
const int CUBEMAP_FACES_CUDA = 6;
const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper vector math
__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator*(float b, float3 a) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator/(float3 a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline void operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__device__ inline void operator/=(float3& a, float b) { a.x /= b; a.y /= b; a.z /= b; }

__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ inline float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float3 normalize(float3 v)
{
    float lenSq = dot(v, v);
    if (lenSq < 1e-8f) return make_float3(0, 0, 0);
    float invLen = 1.0f / sqrtf(lenSq);
    return v * invLen;
}

__device__ float3 FaceUVToDir(int face, float u, float v)
{
    float3 dir;
    switch (face)
    {
        case 0: dir = make_float3(1.0f, v, -u); break;  // +X
        case 1: dir = make_float3(-1.0f, v, u); break;  // -X
        case 2: dir = make_float3(u, 1.0f, -v); break;  // +Y
        case 3: dir = make_float3(u, -1.0f, v); break;  // -Y
        case 4: dir = make_float3(u, v, 1.0f); break;   // +Z
        case 5: dir = make_float3(-u, v, -1.0f); break; // -Z
        default: dir = make_float3(0,0,0); break;
    }
    return normalize(dir);
}

__device__ float2 Hammersley(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = float(bits) * 2.3283064365386963e-10f;
    return make_float2(float(i) / float(N), rdi);
}

__device__ float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = CUDART_PI_F * denom * denom;
    return nom / denom;
}

__device__ float3 ImportanceSampleGGX(float2 Xi, float roughness, float3 N)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float phi = 2.0f * CUDART_PI_F * Xi.x;
    float cosTheta = sqrtf(fmaxf(0.0f, (1.0f - Xi.y) / (1.0f + (a2 - 1.0f) * Xi.y)));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    
    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    
    return normalize(T * H.x + B * H.y + N * H.z);
}

__device__ float GeometrySchlickGGX(float NdotV, float roughness)
{
	float a = roughness;
	float k = (a * a) / 2.0f;

	float nom = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return nom / denom;
}

__device__ float GeometrySmith(float roughness, float NoV, float NoL)
{
	float ggx2 = GeometrySchlickGGX(NoV, roughness);
	float ggx1 = GeometrySchlickGGX(NoL, roughness);

	return ggx1 * ggx2;
}

// Global kernels
__global__ void BakeIrradianceKernel(cudaTextureObject_t envMap, float* dst, int width, int height, int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;

    if (x >= width || y >= height) return;

    float u = (2.0f * (x + 0.5f) / width) - 1.0f;
    float v = 1.0f - (2.0f * (y + 0.5f) / height);
    float3 Ndir = FaceUVToDir(face, u, v);
    
    float3 irradiance = make_float3(0, 0, 0);

    float3 up = fabsf(Ndir.y) < 0.999f ? make_float3(0, 1, 0) : make_float3(0, 0, 1);
    float3 right = normalize(cross(up, Ndir));
    up = normalize(cross(Ndir, right));

    float sampleDelta = (2.0f * CUDART_PI_F) / sampleCount;
    float nrSamples = 0.0f;
    for (float phi = 0.0f; phi < 2.0f * CUDART_PI_F; phi += sampleDelta)
    {
        for (float theta = 0.0f; theta < 0.5f * CUDART_PI_F; theta += sampleDelta)
        {
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);
            float3 tangentSample = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
            float3 sampleVec = right * tangentSample.x + up * tangentSample.y + Ndir * tangentSample.z;

            float4 tex = texCubemapLod<float4>(envMap, sampleVec.x, sampleVec.y, sampleVec.z, 0.0f);
            irradiance += make_float3(tex.x, tex.y, tex.z) * cosTheta * sinTheta;
            nrSamples++;
        }
    }
    irradiance = CUDART_PI_F * irradiance * (1.0f / nrSamples);

    int pixelOffset = (face * width * height + y * width + x) * 4;
    dst[pixelOffset + 0] = irradiance.x;
    dst[pixelOffset + 1] = irradiance.y;
    dst[pixelOffset + 2] = irradiance.z;
    dst[pixelOffset + 3] = 1.0f;
}

__global__ void BakeRadianceKernel(cudaTextureObject_t envMap, float* dst, int width, int height, int mip, int totalMips, int sampleCount, float srcWidth, float srcHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;

    if (x >= width || y >= height) return;

    float roughness = (totalMips <= 1) ? 0.0f : (float)mip / (float)(totalMips - 1);

    float u = (2.0f * (x + 0.5f) / width) - 1.0f;
    float v = 1.0f - (2.0f * (y + 0.5f) / height);
    float3 Ndir = FaceUVToDir(face, u, v);

    float3 V = Ndir;
    float3 prefilteredColor = make_float3(0, 0, 0);
    float totalWeight = 0.0f;

    for (uint32_t i = 0; i < sampleCount; ++i) 
    {
        float2 Xi = Hammersley(i, sampleCount);
        float3 H = ImportanceSampleGGX(Xi, roughness, Ndir);
        float3 L = normalize(2.0f * dot(V, H) * H - V);

        float NdotL = fmaxf(dot(Ndir, L), 0.0f);
        if (NdotL > 0.0f) 
        {
            float NdotH = fmaxf(dot(Ndir, H), 0.0f);
            float VdotH = fmaxf(dot(V, H), 0.0f);
            float D = DistributionGGX(NdotH, roughness);
            float pdf = (D * NdotH / (4.0f * VdotH)) + 0.0001f;

            float saTexel = 4.0f * CUDART_PI_F / (6.0f * srcWidth * srcHeight);
            float saSample = 1.0f / (float(sampleCount) * pdf + 0.0001f);
            float mipSource = (roughness == 0.0f) ? 0.0f : 0.5f * log2f(saSample / saTexel);

            float4 tex = texCubemapLod<float4>(envMap, L.x, L.y, L.z, mipSource);
            prefilteredColor += make_float3(tex.x, tex.y, tex.z) * NdotL;
            totalWeight += NdotL;
        }
    }

    if (totalWeight > 0.0f) prefilteredColor /= totalWeight;

    int pixelOffset = (face * width * height + y * width + x) * 4;
    dst[pixelOffset + 0] = prefilteredColor.x;
    dst[pixelOffset + 1] = prefilteredColor.y;
    dst[pixelOffset + 2] = prefilteredColor.z;
    dst[pixelOffset + 3] = 1.0f;
}

__global__ void BakeBRDFLUTKernel(float* dst, int width, int height, int sampleCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float NdotV = (x + 0.5f) / (float)width;
    float roughness = 1.0f - (y + 0.5f) / (float)height;

    // Small epsilon to avoid artifacts at 0
    NdotV = fmaxf(NdotV, 0.001f);

    float3 V;
    V.x = sqrtf(1.0f - NdotV * NdotV);
    V.y = 0.0f;
    V.z = NdotV;

    float A = 0.0f;
    float B = 0.0f;

    float3 N = make_float3(0, 0, 1);

    for (uint32_t i = 0u; i < sampleCount; ++i)
    {
        float2 Xi = Hammersley(i, sampleCount);
        float3 H = ImportanceSampleGGX(Xi, roughness, N);
        float3 L = normalize(2.0f * dot(V, H) * H - V);

        float NoL = fmaxf(L.z, 0.0f);
        float NoH = fmaxf(H.z, 0.0f);
        float VoH = fmaxf(dot(V, H), 0.0f);
        float NoV = fmaxf(dot(N, V), 0.0f);

        if (NoL > 0.0f)
        {
            float G = GeometrySmith(roughness, NoV, NoL);

            float G_Vis = (G * VoH) / (NoH * NoV);
            float Fc = powf(1.0f - VoH, 5.0f);

            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    int pixelOffset = (y * width + x) * 4;
    dst[pixelOffset + 0] = A / (float)sampleCount;
    dst[pixelOffset + 1] = B / (float)sampleCount;
    dst[pixelOffset + 2] = 0.0f; // Could be used for something else, but standard is RG16 or similar
    dst[pixelOffset + 3] = 1.0f;
}

// Helper to find the best matching mip level in src that is at least as big as targetW x targetH
int FindBestMip(const TextureData& src, int targetW, int targetH)
{
    int bestMip = 0;
    for (int m = 0; m < src.mipCount; ++m)
    {
        int mw = std::max(1, src.width >> m);
        int mh = std::max(1, src.height >> m);
        if (mw >= targetW && mh >= targetH)
        {
            bestMip = m;
        }
        else
        {
            break;
        }
    }
    return bestMip;
}

// Helper to create a texture object from TextureData struct starting from baseMip (strictly Cubemap)
cudaTextureObject_t CreateTextureCUDA(const TextureData& src, int baseMip, cudaMipmappedArray_t& mipArray)
{
    int startW = std::max(1, src.width >> baseMip);
    int startH = std::max(1, src.height >> baseMip);
    int startMipCount = src.mipCount - baseMip;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    cudaExtent extent = make_cudaExtent(startW, startH, CUBEMAP_FACES_CUDA);
    unsigned int flags = cudaArrayCubemap;
    
    CUDA_CHECK(cudaMallocMipmappedArray(&mipArray, &channelDesc, extent, startMipCount, flags));

    // Calculate initial float offset to the start of baseMip
    size_t floatOffset = 0;
    for (int m = 0; m < baseMip; ++m)
    {
        int mw = std::max(1, src.width >> m);
        int mh = std::max(1, src.height >> m);
        floatOffset += (size_t)mw * mh * CUBEMAP_FACES_CUDA * COMPONENTS_PER_PIXEL_CUDA;
    }

    for (int m = 0; m < startMipCount; ++m)
    {
        int mipW = std::max(1, startW >> m);
        int mipH = std::max(1, startH >> m);
        cudaArray_t levelArray;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, mipArray, m));

        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)&src.data[floatOffset], mipW * 4 * sizeof(float), mipW, mipH);
        copyParams.dstArray = levelArray;
        copyParams.extent = make_cudaExtent(mipW, mipH, CUBEMAP_FACES_CUDA);
        copyParams.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));

        floatOffset += (size_t)mipW * mipH * CUBEMAP_FACES_CUDA * COMPONENTS_PER_PIXEL_CUDA;
    }

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = mipArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

void BakeIrradianceCUDA(const TextureData& src, TextureData& dst, int sampleCount)
{
    int baseMip = FindBestMip(src, dst.width, dst.height);
    
    cudaMipmappedArray_t envMipArray;
    cudaTextureObject_t envMap = CreateTextureCUDA(src, baseMip, envMipArray);

    float* d_dst;
    size_t dstSize = (size_t)dst.width * dst.height * dst.numFaces * COMPONENTS_PER_PIXEL_CUDA * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dst, dstSize));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid((dst.width + block.x - 1) / block.x, (dst.height + block.y - 1) / block.y, dst.numFaces);

    BakeIrradianceKernel<<<grid, block>>>(envMap, d_dst, dst.width, dst.height, sampleCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.data.data(), d_dst, dstSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaDestroyTextureObject(envMap));
    CUDA_CHECK(cudaFreeMipmappedArray(envMipArray));
}

void BakeRadianceCUDA(const TextureData& src, TextureData& dst, int sampleCount)
{
    float* d_dst;
    size_t totalBytes = dst.data.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dst, totalBytes));

    size_t offset = 0;
    for (int mip = 0; mip < dst.mipCount; ++mip)
    {
        int mipW = std::max(1, dst.width >> mip);
        int mipH = std::max(1, dst.height >> mip);
        
        int baseMip = FindBestMip(src, mipW, mipH);
        int startW = std::max(1, src.width >> baseMip);
        int startH = std::max(1, src.height >> baseMip);

        cudaMipmappedArray_t envMipArray;
        cudaTextureObject_t envMap = CreateTextureCUDA(src, baseMip, envMipArray);

        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((mipW + block.x - 1) / block.x, (mipH + block.y - 1) / block.y, dst.numFaces);

        BakeRadianceKernel<<<grid, block>>>(envMap, d_dst + offset, mipW, mipH, mip, dst.mipCount, sampleCount, (float)startW, (float)startH);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaDestroyTextureObject(envMap));
        CUDA_CHECK(cudaFreeMipmappedArray(envMipArray));

        offset += (size_t)mipW * mipH * dst.numFaces * 4;
    }

    CUDA_CHECK(cudaMemcpy(dst.data.data(), d_dst, totalBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dst));
}

void BakeBRDFLUT_CUDA(TextureData& dst, int sampleCount)
{
    float* d_dst;
    size_t dstSize = (size_t)dst.width * dst.height * COMPONENTS_PER_PIXEL_CUDA * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dst, dstSize));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid((dst.width + block.x - 1) / block.x, (dst.height + block.y - 1) / block.y, 1);

    BakeBRDFLUTKernel<<<grid, block>>>(d_dst, dst.width, dst.height, sampleCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.data.data(), d_dst, dstSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dst));
}
