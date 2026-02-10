#define NOMINMAX

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "bake_ibl.h"

#include <DirectXTex.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_HDR
#include "stb_image.h"

using namespace DirectX;

// Constants
const int COMPONENTS_PER_PIXEL = 4; // RGBA
const int BYTES_PER_FLOAT = 4;
const int BYTES_PER_PIXEL = COMPONENTS_PER_PIXEL * BYTES_PER_FLOAT; // 16
const int CUBEMAP_FACES = 6;
const int DEFAULT_IRR_SIZE = 64;
const int DEFAULT_RAD_SIZE = 256;
const int DEFAULT_IRR_SAMPLES = 1024;
const int DEFAULT_RAD_SAMPLES = 8192;
const int DEFAULT_BRDF_SAMPLES = 1024;
const int DEFAULT_BRDF_SIZE = 256;
const int MAX_INPUT_SIZE = 1024;

char* HRESULT_To_String(const HRESULT hr)
{
    static char buffer[512];
    DWORD code = HRESULT_CODE(hr); // Extract Win32 error code

    if (FormatMessage(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buffer,
        sizeof(buffer),
        NULL
    ) == 0)
    {
        // Fallback if FormatMessage fails
        sprintf_s(buffer, sizeof(buffer), "Unknown error code: 0x%08X", hr);
    }

    return buffer;
}

XMVECTOR FaceUVToDir(int face, float u, float v)
{
    XMVECTOR dir;
    switch (face)
    {
    case 0: dir = XMVectorSet(1.0f, v, -u, 0.0f); break;  // +X
    case 1: dir = XMVectorSet(-1.0f, v, u, 0.0f); break;  // -X
    case 2: dir = XMVectorSet(u, 1.0f, -v, 0.0f); break;  // +Y
    case 3: dir = XMVectorSet(u, -1.0f, v, 0.0f); break;  // -Y
    case 4: dir = XMVectorSet(u, v, 1.0f, 0.0f); break;   // +Z
    case 5: dir = XMVectorSet(-u, v, -1.0f, 0.0f); break; // -Z
    default: dir = XMVectorZero(); break;
    }
    return XMVector3Normalize(dir);
}

ScratchImage ConvertEquirectangularToCubemap(const ScratchImage& equirect)
{
    const TexMetadata& eqMeta = equirect.GetMetadata();
    size_t faceSize = eqMeta.height; 
    
    printf("Converting equirectangular (%zux%zu) to cubemap (%zux%zu)...\n", eqMeta.width, eqMeta.height, faceSize, faceSize);

    TexMetadata cubeMeta = {};
    cubeMeta.width = faceSize;
    cubeMeta.height = faceSize;
    cubeMeta.depth = 1;
    cubeMeta.arraySize = 6;
    cubeMeta.mipLevels = 1;
    cubeMeta.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    cubeMeta.dimension = TEX_DIMENSION_TEXTURE2D;
    cubeMeta.miscFlags = TEX_MISC_TEXTURECUBE;

    ScratchImage cube;
    if (HRESULT result = cube.Initialize(cubeMeta);
        FAILED(result))
    {
        printf("Error: Failed to initialize ScratchImage. %s\n", HRESULT_To_String(result));
        return {};
    }

    const Image* eqImg = equirect.GetImage(0, 0, 0);

    for (int face = 0; face < 6; ++face)
    {
        const Image* faceImg = cube.GetImage(0, face, 0);
        float* pixels = (float*)faceImg->pixels;

        for (size_t y = 0; y < faceSize; ++y)
        {
            for (size_t x = 0; x < faceSize; ++x)
            {
                float u = (2.0f * (x + 0.5f) / (float)faceSize) - 1.0f;
                float v = 1.0f - (2.0f * (y + 0.5f) / (float)faceSize);

                XMVECTOR dir = FaceUVToDir(face, u, v);
                XMFLOAT3 dirF3;
                XMStoreFloat3(&dirF3, dir);

                float phi = atan2f(dirF3.x, dirF3.z);
                float theta = acosf(std::max(-1.0f, std::min(1.0f, dirF3.y)));

                float eqU = (phi / (2.0f * XM_PI)) + 0.5f;
                float eqV = theta / XM_PI;

                int tx = std::max(0, std::min((int)eqMeta.width - 1, (int)(eqU * eqMeta.width)));
                int ty = std::max(0, std::min((int)eqMeta.height - 1, (int)(eqV * eqMeta.height)));

                const float* srcPixel = (const float*)(eqImg->pixels + (ty * eqImg->rowPitch) + (tx * 16));
                float* dstPixel = pixels + (y * faceSize + x) * 4;
                memcpy(dstPixel, srcPixel, 16);
            }
        }
    }

    return cube;
}

bool SetupTextureDataFromImage(ScratchImage& processed, TextureData& td, bool isCubemap)
{
    // Performance Optimization: If the input is extremely large (e.g. 4K), downsample it to something manageable for IBL baking.
    if (processed.GetMetadata().width > MAX_INPUT_SIZE || processed.GetMetadata().height > MAX_INPUT_SIZE)
    {
        float aspect = (float)processed.GetMetadata().width / (float)processed.GetMetadata().height;
        size_t targetW, targetH;
        if (aspect >= 1.0f)
        {
            targetW = MAX_INPUT_SIZE;
            targetH = (size_t)((float)MAX_INPUT_SIZE / aspect);
        }
        else
        {
            targetH = MAX_INPUT_SIZE;
            targetW = (size_t)((float)MAX_INPUT_SIZE * aspect);
        }

        printf("Input is large (%zux%zu). Resizing to %zux%zu for baking...\n",
            processed.GetMetadata().width, processed.GetMetadata().height, targetW, targetH);

        ScratchImage resized;
        if (SUCCEEDED(Resize(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(),
            targetW, targetH, TEX_FILTER_LINEAR, resized)))
        {
            processed = std::move(resized);
        }
    }

    // Convert equirectangular to cubemap if necessary
    if (!isCubemap)
    {
        ScratchImage cube = ConvertEquirectangularToCubemap(processed);
        if (cube.GetImageCount() > 0)
        {
            processed = std::move(cube);
            isCubemap = true; // Now it's a cubemap
        }
        else
        {
            printf("Error: Failed to convert equirectangular to cubemap.\n");
            return false;
        }
    }

    // Generate a full mip chain for the source texture if it's missing or after resize.
    if (processed.GetMetadata().mipLevels <= 1)
    {
        printf("Generating mipmaps for source texture...\n");
        ScratchImage mips;
        if (SUCCEEDED(GenerateMipMaps(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), TEX_FILTER_LINEAR, 0, mips)))
        {
            processed = std::move(mips);
        }
        else
        {
            printf("Error: Failed to generate mipmaps.\n");
            return false;
        }
    }

    td.width = (int)processed.GetMetadata().width;
    td.height = (int)processed.GetMetadata().height;
    td.mipCount = (int)processed.GetMetadata().mipLevels;
    td.numFaces = CUBEMAP_FACES;

    size_t totalFloats = 0;
    for (int m = 0; m < td.mipCount; ++m)
    {
        int mipW = std::max(1, td.width >> m);
        int mipH = std::max(1, td.height >> m);
        totalFloats += (size_t)mipW * mipH * td.numFaces * COMPONENTS_PER_PIXEL;
    }
    td.data.resize(totalFloats);

    size_t floatOffset = 0;
    for (int m = 0; m < td.mipCount; ++m)
    {
        int mipW = std::max(1, td.width >> m);
        int mipH = std::max(1, td.height >> m);
        for (int face = 0; face < td.numFaces; ++face)
        {
            const Image* img = processed.GetImage(m, face, 0);
            if (!img) return false;
            memcpy(&td.data[floatOffset], img->pixels, (size_t)mipW * mipH * BYTES_PER_PIXEL);
            floatOffset += (size_t)mipW * mipH * COMPONENTS_PER_PIXEL;
        }
    }

    return true;
}

bool LoadDDSTexture(std::string_view path, TextureData& td)
{
    std::filesystem::path p(path);
    std::wstring wpath = p.wstring();
    TexMetadata metadata;
    ScratchImage image;

    printf("Loading DDS file: %.*s\n", (int)path.size(), path.data());

    if (HRESULT result = LoadFromDDSFile(wpath.c_str(), DDS_FLAGS_NONE, &metadata, image);
        FAILED(result))
    {
        printf("Error: Failed to load DDS file. %s\n", HRESULT_To_String(result));
        return false;
    }

    // Allow both cubemaps and 2D textures
    bool isCubemap = metadata.IsCubemap();
    if (!isCubemap && metadata.dimension != TEX_DIMENSION_TEXTURE2D)
    {
        printf("Error: Input must be a cubemap or 2D texture.\n");
        return false;
    }

    ScratchImage processed;

    if (IsCompressed(metadata.format))
    {
        if (HRESULT result = Decompress(image.GetImages(), image.GetImageCount(), metadata, DXGI_FORMAT_R32G32B32A32_FLOAT, processed);
            FAILED(result))
        {
            printf("Error: Failed to decompress DDS file. %s\n", HRESULT_To_String(result));
            return false;
        }
    }
    else if (metadata.format != DXGI_FORMAT_R32G32B32A32_FLOAT)
    {
        if (HRESULT result = Convert(image.GetImages(), image.GetImageCount(), metadata, DXGI_FORMAT_R32G32B32A32_FLOAT, TEX_FILTER_LINEAR, TEX_THRESHOLD_DEFAULT, processed);
            FAILED(result))
        {
            printf("Error: Failed to convert DDS file. %s\n", HRESULT_To_String(result));
            return false;
        }
    }
    else
    {
        processed = std::move(image);
    }

    return SetupTextureDataFromImage(processed, td, isCubemap);
}

bool LoadHDRTexture(std::string_view path, TextureData& td)
{
    printf("Loading HDR file: %.*s\n", (int)path.size(), path.data());

    if (!stbi_is_hdr(path.data()))
    {
        printf("Error: File is not a valid HDR image.\n");
        return false;
    }

    int width, height, channels;
    float* data = stbi_loadf(path.data(), &width, &height, &channels, 4);
    if (!data)
    {
        printf("Error: Failed to load HDR file with stb_image: %s\n", stbi_failure_reason());
        return false;
    }

    ScratchImage image;
    if (HRESULT result = image.Initialize2D(DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, 1, 1);
        FAILED(result))
    {
        printf("Error: Failed to initialize ScratchImage for HDR. %s\n", HRESULT_To_String(result));
        stbi_image_free(data);
        return false;
    }

    memcpy(image.GetPixels(), data, (size_t)width * height * 16);
    stbi_image_free(data);

    // HDR files loaded via stbi are always 2D equirectangular in this context
    return SetupTextureDataFromImage(image, td, false);
}

void SaveDDS(std::string_view path, const TextureData& td, DXGI_FORMAT targetFormat = DXGI_FORMAT_UNKNOWN)
{
    std::filesystem::path p(path);
    std::wstring wpath = p.wstring();
    
    bool isCubemap = (td.numFaces == 6);

    TexMetadata metadata = {};
    metadata.width = td.width;
    metadata.height = td.height;
    metadata.depth = 1;
    metadata.arraySize = td.numFaces;
    metadata.mipLevels = td.mipCount;
    metadata.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    metadata.dimension = TEX_DIMENSION_TEXTURE2D;
    metadata.miscFlags = isCubemap ? TEX_MISC_TEXTURECUBE : 0;

    ScratchImage image;
    if (FAILED(image.Initialize(metadata)))
    {
        printf("Error: Failed to initialize ScratchImage for saving.\n");
        return;
    }

    size_t floatOffset = 0;
    for (int m = 0; m < td.mipCount; ++m)
    {
        int mipW = std::max(1, td.width >> m);
        int mipH = std::max(1, td.height >> m);
        for (int face = 0; face < td.numFaces; ++face)
        {
            const Image* img = image.GetImage(m, face, 0);
            memcpy(img->pixels, &td.data[floatOffset], (size_t)mipW * mipH * BYTES_PER_PIXEL);
            floatOffset += (size_t)mipW * mipH * COMPONENTS_PER_PIXEL;
        }
    }

    if (targetFormat != DXGI_FORMAT_UNKNOWN)
    {
        ScratchImage converted;
        if (SUCCEEDED(Convert(image.GetImages(), image.GetImageCount(), image.GetMetadata(), targetFormat, TEX_FILTER_DEFAULT, TEX_THRESHOLD_DEFAULT, converted)))
        {
            SaveToDDSFile(converted.GetImages(), converted.GetImageCount(), converted.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
            return;
        }
        else
        {
            printf("Warning: Failed to convert to requested format %d. Falling back to default.\n", (int)targetFormat);
        }
    }

    ScratchImage processed;
    if (isCubemap && SUCCEEDED(Compress(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DXGI_FORMAT_BC6H_UF16, TEX_COMPRESS_PARALLEL, TEX_THRESHOLD_DEFAULT, processed)))
    {
        SaveToDDSFile(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
    }
    else
    {
        if (isCubemap) printf("Warning: Failed to compress to BC6H_UF16. Saving as R32G32B32A32_FLOAT.\n");
        SaveToDDSFile(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
    }
}

int main(int argc, char** argv)
{
    // Check for CUDA toolkit
    if (system("nvcc --version >nul 2>&1") != 0)
    {
        printf("Error: CUDA toolkit not found. Please install CUDA and ensure nvcc is in PATH.\n");
        return 1;
    }

    std::filesystem::path inputFile;
    bool showHelp = false;
    int irrSize = DEFAULT_IRR_SIZE;
    int radSize = DEFAULT_RAD_SIZE;
    int irrSamples = DEFAULT_IRR_SAMPLES;
    int radSamples = DEFAULT_RAD_SAMPLES;
    bool bakeBrdf = false;
    int brdfSize = DEFAULT_BRDF_SIZE;
    int brdfSamples = DEFAULT_BRDF_SAMPLES;
    bool brdfOnly = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            showHelp = true;
            break;
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            irrSize = std::atoi(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc)
        {
            radSize = std::atoi(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            irrSamples = std::atoi(argv[++i]);
        }
        else if (arg == "-m" && i + 1 < argc)
        {
            radSamples = std::atoi(argv[++i]);
        }
        else if ((arg == "-b" || arg == "--brdf"))
        {
            bakeBrdf = true;
        }
        else if (arg == "--brdf-only")
        {
            bakeBrdf = true;
            brdfOnly = true;
        }
        else if (arg == "-l" && i + 1 < argc)
        {
            brdfSize = std::atoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc)
        {
            brdfSamples = std::atoi(argv[++i]);
        }
        else if (inputFile.empty())
        {
            inputFile = std::filesystem::path(argv[i]);
        }
        else
        {
            printf("Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (showHelp || (inputFile.empty() && !bakeBrdf))
    {
        printf("Usage: DDSToIBL <input.dds/.hdr> [options]\n");
        printf("Convert DDS cubemap or 2D (equirectangular) texture to IBL irradiance and radiance cubemaps.\n");
        printf("Supported formats: All DDS formats and HDR.\n");
        printf("Baking is performed on CUDA.\n");
        printf("Default irradiance cubemap face size: %u\n", DEFAULT_IRR_SIZE);
        printf("Default radiance cubemap face size: %u\n", DEFAULT_RAD_SIZE);
        printf("Default irradiance samples: %u\n", DEFAULT_IRR_SAMPLES);
        printf("Default radiance samples: %u\n", DEFAULT_RAD_SAMPLES);
        printf("Default BRDF samples: %u\n", DEFAULT_BRDF_SAMPLES);
        printf("\n");
        printf("Options:\n");
        printf("  -h, --help             Show this help message\n");
        printf("  -i <size>              Irradiance cubemap face size (default: %u)\n", DEFAULT_IRR_SIZE);
        printf("  -r <size>              Radiance cubemap face size (default: %u)\n", DEFAULT_RAD_SIZE);
        printf("  -s <count>             Number of samples for irradiance baking (default: %u)\n", DEFAULT_IRR_SAMPLES);
        printf("  -m <count>             Number of samples for radiance baking (default: %u)\n", DEFAULT_RAD_SAMPLES);
        printf("  -b, --brdf             Bake BRDF LUT\n");
        printf("  -l <size>              BRDF LUT size (default: %u)\n", DEFAULT_BRDF_SIZE);
        printf("  -n <count>             Number of samples for BRDF LUT baking (default: %u)\n", DEFAULT_BRDF_SAMPLES);
        printf("  --brdf-only            Only bake BRDF LUT (ignores input texture)\n");
        return showHelp ? 0 : 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (!brdfOnly && !inputFile.empty())
    {
        TextureData env;
        bool loaded = false;
        
        std::string ext = inputFile.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".hdr")
        {
            loaded = LoadHDRTexture(inputFile.string(), env);
        }
        else
        {
            loaded = LoadDDSTexture(inputFile.string(), env);
        }

        if (!loaded)
        {
            printf("Failed to load %s\n", inputFile.string().c_str());
            return 1;
        }

        printf("Loaded %s: %dx%d, mips %d\n", (env.numFaces == 6) ? "cubemap" : "texture", env.width, env.height, env.mipCount);

        TextureData irradiance;
        irradiance.width = irrSize;
        irradiance.height = irrSize;
        irradiance.mipCount = 1;
        irradiance.numFaces = env.numFaces;
        irradiance.data.resize((size_t)irradiance.width * irradiance.height * irradiance.numFaces * COMPONENTS_PER_PIXEL);

        TextureData radiance;
        radiance.width = radSize;
        radiance.height = radSize;
        radiance.mipCount = 0;
        radiance.numFaces = env.numFaces;
        int s = radSize;
        while (s > 0)
        {
            radiance.mipCount++;
            s /= 2;
        }

        size_t totalFloats = 0;
        for (int m = 0; m < radiance.mipCount; ++m)
        {
            int mipW = std::max(1, radiance.width >> m);
            int mipH = std::max(1, radiance.height >> m);
            totalFloats += (size_t)mipW * mipH * radiance.numFaces * COMPONENTS_PER_PIXEL;
        }
        radiance.data.resize(totalFloats);

        printf("Baking Irradiance (%dx%d, %d samples)...\n", irrSize, irrSize, irrSamples);
        BakeIrradianceCUDA(env, irradiance, irrSamples);
        printf("Baking Radiance (%dx%d, %d samples)...\n", radSize, radSize, radSamples);
        BakeRadianceCUDA(env, radiance, radSamples);

        // Generate output filenames
        std::filesystem::path baseFilename = inputFile;
        baseFilename.replace_extension("");

        std::filesystem::path irrOutput = baseFilename;
        irrOutput += "_irradiance.dds";
        std::filesystem::path radOutput = baseFilename;
        radOutput += "_radiance.dds";

        printf("Saving %s...\n", irrOutput.string().c_str());
        SaveDDS(irrOutput.string(), irradiance);
        printf("Saving %s...\n", radOutput.string().c_str());
        SaveDDS(radOutput.string(), radiance);
    }

    if (bakeBrdf)
    {
        TextureData brdf;
        brdf.width = brdfSize;
        brdf.height = brdfSize;
        brdf.mipCount = 1;
        brdf.numFaces = 1;
        brdf.data.resize((size_t)brdf.width * brdf.height * COMPONENTS_PER_PIXEL);
        printf("Baking BRDF LUT (%dx%d, %d samples)...\n", brdfSize, brdfSize, brdfSamples);
        BakeBRDFLUT_CUDA(brdf, brdfSamples);

        std::filesystem::path brdfOutput = inputFile.empty() ? "brdf_lut.dds" : inputFile.parent_path() / "brdf_lut.dds";
        printf("Saving %s...\n", brdfOutput.string().c_str());
        SaveDDS(brdfOutput.string(), brdf, DXGI_FORMAT_R16G16_FLOAT);
    }

    printf("Done.\n");

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    printf("Total runtime: %.2f seconds\n", elapsed.count());

    return 0;
}
