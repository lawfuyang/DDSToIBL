#include "bake_ibl.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

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
    float invLen = 1.0f / sqrtf(dot(v, v));
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

// Global kernels
__global__ void BakeIrradianceKernel(cudaTextureObject_t envMap, float* dst, int size, bool isCubemap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;

    if (x >= size || y >= size) return;

    float3 Ndir;
    if (isCubemap)
    {
        float u = (2.0f * (x + 0.5f) / size) - 1.0f;
        float v = 1.0f - (2.0f * (y + 0.5f) / size);
        Ndir = FaceUVToDir(face, u, v);
    }
    else
    {
        // For 2D texture, assume equirectangular
        float u = (x + 0.5f) / size;
        float v = (y + 0.5f) / size;
        float theta = v * CUDART_PI_F; // polar angle
        float phi = u * 2.0f * CUDART_PI_F; // azimuthal angle
        Ndir = make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
    }
    float3 irradiance = make_float3(0, 0, 0);

    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(up, Ndir));
    up = normalize(cross(Ndir, right));

    float sampleDelta = 0.0025f;
    float nrSamples = 0.0f;
    for (float phi = 0.0f; phi < 2.0f * CUDART_PI_F; phi += sampleDelta)
    {
        for (float theta = 0.0f; theta < 0.5f * CUDART_PI_F; theta += sampleDelta)
        {
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);
            float3 tangentSample = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
            float3 sampleVec = right * tangentSample.x + up * tangentSample.y + Ndir * tangentSample.z;

            float4 tex;
            if (isCubemap)
            {
                tex = texCubemapLod<float4>(envMap, sampleVec.x, sampleVec.y, sampleVec.z, 0.0f);
            }
            else
            {
                // For 2D equirectangular
                float u = atan2f(sampleVec.x, sampleVec.z) / (2.0f * CUDART_PI_F) + 0.5f;
                float v = acosf(fmaxf(-1.0f, fminf(1.0f, sampleVec.y))) / CUDART_PI_F;
                tex = tex2DLod<float4>(envMap, u, v, 0.0f);
            }
            irradiance += make_float3(tex.x, tex.y, tex.z) * cosTheta * sinTheta;
            nrSamples++;
        }
    }
    irradiance = CUDART_PI_F * irradiance * (1.0f / nrSamples);

    int pixelOffset = (face * size * size + y * size + x) * 4;
    dst[pixelOffset + 0] = irradiance.x;
    dst[pixelOffset + 1] = irradiance.y;
    dst[pixelOffset + 2] = irradiance.z;
    dst[pixelOffset + 3] = 1.0f;
}

__global__ void BakeRadianceKernel(cudaTextureObject_t envMap, float* dst, int size, int mip, int totalMips, int sampleCount, float srcSize, bool isCubemap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;

    if (x >= size || y >= size) return;

    float roughness = (totalMips <= 1) ? 0.0f : (float)mip / (float)(totalMips - 1);

    float3 Ndir;
    if (isCubemap)
    {
        float u = (2.0f * (x + 0.5f) / size) - 1.0f;
        float v = 1.0f - (2.0f * (y + 0.5f) / size);
        Ndir = FaceUVToDir(face, u, v);
    }
    else
    {
        // For 2D texture, assume equirectangular
        float u = (x + 0.5f) / size;
        float v = (y + 0.5f) / size;
        float theta = v * CUDART_PI_F; // polar angle
        float phi = u * 2.0f * CUDART_PI_F; // azimuthal angle
        Ndir = make_float3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi));
    }
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

            float saTexel = 4.0f * CUDART_PI_F / (float(isCubemap ? 6 : 1) * srcSize * srcSize);
            float saSample = 1.0f / (float(sampleCount) * pdf + 0.0001f);
            float mipSource = (roughness == 0.0f) ? 0.0f : 0.5f * log2f(saSample / saTexel);

            float4 tex;
            if (isCubemap)
            {
                tex = texCubemapLod<float4>(envMap, L.x, L.y, L.z, mipSource);
            }
            else
            {
                // For 2D equirectangular
                float u = atan2f(L.x, L.z) / (2.0f * CUDART_PI_F) + 0.5f;
                float v = acosf(fmaxf(-1.0f, fminf(1.0f, L.y))) / CUDART_PI_F;
                tex = tex2DLod<float4>(envMap, u, v, mipSource);
            }
            prefilteredColor += make_float3(tex.x, tex.y, tex.z) * NdotL;
            totalWeight += NdotL;
        }
    }

    if (totalWeight > 0.0f) prefilteredColor /= totalWeight;

    // Output offset calculation needs to account for previous mips if dst is a single pointer
    // But we'll pass the pointer to the start of the current mip for simplicity
    int pixelOffset = (face * size * size + y * size + x) * 4;
    dst[pixelOffset + 0] = prefilteredColor.x;
    dst[pixelOffset + 1] = prefilteredColor.y;
    dst[pixelOffset + 2] = prefilteredColor.z;
    dst[pixelOffset + 3] = 1.0f;
}

// Function to create a cubemap texture object from Cubemap struct
cudaTextureObject_t CreateCubemapTexture(const Cubemap& src, cudaMipmappedArray_t& mipArray)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    cudaExtent extent = make_cudaExtent(src.size, src.size, CUBEMAP_FACES_CUDA);
    CUDA_CHECK(cudaMallocMipmappedArray(&mipArray, &channelDesc, extent, src.mipCount, cudaArrayCubemap));

    int floatOffset = 0;
    for (int m = 0; m < src.mipCount; ++m)
    {
        int mipSize = std::max(1, src.size >> m);
        cudaArray_t levelArray;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, mipArray, m));

        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)&src.data[floatOffset], mipSize * 4 * sizeof(float), mipSize, mipSize);
        copyParams.dstArray = levelArray;
        copyParams.extent = make_cudaExtent(mipSize, mipSize, CUBEMAP_FACES_CUDA);
        copyParams.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));

        floatOffset += mipSize * mipSize * CUBEMAP_FACES_CUDA * COMPONENTS_PER_PIXEL_CUDA;
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
    texDesc.normalizedCoords = 0; // Use direction vectors for cubemaps
    
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

// Function to create a 2D texture object from Cubemap struct (for 2D textures)
cudaTextureObject_t CreateTexture2D(const Cubemap& src, cudaMipmappedArray_t& mipArray)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    cudaExtent extent = make_cudaExtent(src.size, src.size, 1);
    CUDA_CHECK(cudaMallocMipmappedArray(&mipArray, &channelDesc, extent, src.mipCount, 0));

    int floatOffset = 0;
    for (int m = 0; m < src.mipCount; ++m)
    {
        int mipSize = std::max(1, src.size >> m);
        cudaArray_t levelArray;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, mipArray, m));

        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)&src.data[floatOffset], mipSize * 4 * sizeof(float), mipSize, mipSize);
        copyParams.dstArray = levelArray;
        copyParams.extent = make_cudaExtent(mipSize, mipSize, 1);
        copyParams.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));

        floatOffset += mipSize * mipSize * 4;
    }

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = mipArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // Use normalized coords for 2D
    
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

void BakeIrradianceCUDA(const Cubemap& src, Cubemap& dst, int numFaces)
{
    cudaMipmappedArray_t envMipArray;
    cudaTextureObject_t envMap;
    if (numFaces == 6)
    {
        envMap = CreateCubemapTexture(src, envMipArray);
    }
    else
    {
        envMap = CreateTexture2D(src, envMipArray);
    }

    float* d_dst;
    size_t dstSize = dst.size * dst.size * numFaces * COMPONENTS_PER_PIXEL_CUDA * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dst, dstSize));

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid((dst.size + block.x - 1) / block.x, (dst.size + block.y - 1) / block.y, numFaces);

    BakeIrradianceKernel<<<grid, block>>>(envMap, d_dst, dst.size, numFaces == 6);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst.data.data(), d_dst, dstSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaDestroyTextureObject(envMap));
    CUDA_CHECK(cudaFreeMipmappedArray(envMipArray));
}

void BakeRadianceCUDA(const Cubemap& src, Cubemap& dst, int numFaces)
{
    const int sampleCount = 2048;
    cudaMipmappedArray_t envMipArray;
    cudaTextureObject_t envMap;
    if (numFaces == 6)
    {
        envMap = CreateCubemapTexture(src, envMipArray);
    }
    else
    {
        envMap = CreateTexture2D(src, envMipArray);
    }

    int totalFloats = (int)dst.data.size();
    float* d_dst;
    CUDA_CHECK(cudaMalloc(&d_dst, totalFloats * sizeof(float)));

    int offset = 0;
    for (int mip = 0; mip < dst.mipCount; ++mip)
    {
        int mipSize = std::max(1, dst.size >> mip);
        
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((mipSize + block.x - 1) / block.x, (mipSize + block.y - 1) / block.y, numFaces);

        BakeRadianceKernel<<<grid, block>>>(envMap, d_dst + offset, mipSize, mip, dst.mipCount, sampleCount, (float)src.size, numFaces == 6);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        offset += mipSize * mipSize * numFaces * 4;
    }

    CUDA_CHECK(cudaMemcpy(dst.data.data(), d_dst, totalFloats * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaDestroyTextureObject(envMap));
    CUDA_CHECK(cudaFreeMipmappedArray(envMipArray));
}
