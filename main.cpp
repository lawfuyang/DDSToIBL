#define NOMINMAX

#include <cassert>
#include <cstring>
#include <filesystem>
#include <string_view>

#include "bake_ibl.h"

#include <DirectXTex.h>

using namespace DirectX;

// Constants
const int COMPONENTS_PER_PIXEL = 4; // RGBA
const int BYTES_PER_FLOAT = 4;
const int BYTES_PER_PIXEL = COMPONENTS_PER_PIXEL * BYTES_PER_FLOAT; // 16
const int CUBEMAP_FACES = 6;
const int DEFAULT_IRR_SIZE = 64;
const int DEFAULT_RAD_SIZE = 256;
const int MAX_INPUT_SIZE = 1024;

bool LoadDDSTexture(std::string_view path, Cubemap& cm, bool& isCubemapOut)
{
    std::filesystem::path p(path);
    std::wstring wpath = p.wstring();
    TexMetadata metadata;
    ScratchImage image;

    printf("Loading DDS file: %.*s\n", (int)path.size(), path.data());

    if (FAILED(LoadFromDDSFile(wpath.c_str(), DDS_FLAGS_NONE, &metadata, image)))
    {
        printf("Error: Failed to load DDS file.\n");
        assert(false && "Failed to load DDS file");
        return false;
    }

    // Allow both cubemaps and 2D textures
    bool isCubemap = metadata.IsCubemap();
    isCubemapOut = isCubemap;
    if (!isCubemap && metadata.dimension != TEX_DIMENSION_TEXTURE2D)
    {
        printf("Error: Input must be a cubemap or 2D texture.\n");
        assert(false && "Input must be a cubemap or 2D texture");
        return false;
    }

    ScratchImage processed;
    if (IsCompressed(metadata.format))
    {
        if (FAILED(Decompress(image.GetImages(), image.GetImageCount(), metadata, DXGI_FORMAT_R32G32B32A32_FLOAT, processed)))
        {
            printf("Error: Failed to decompress DDS file.\n");
            assert(false && "Failed to decompress DDS file");
            return false;
        }
    }
    else if (metadata.format != DXGI_FORMAT_R32G32B32A32_FLOAT)
    {
        if (FAILED(Convert(image.GetImages(), image.GetImageCount(), metadata, DXGI_FORMAT_R32G32B32A32_FLOAT, TEX_FILTER_DEFAULT, TEX_THRESHOLD_DEFAULT, processed)))
        {
            printf("Error: Failed to convert DDS file.\n");
            assert(false && "Failed to convert DDS file");
            return false;
        }
    }
    else
    {
        processed = std::move(image);
    }

    // Performance Optimization: If the input cubemap is extremely large (e.g. 4K), 
    // downsample it to something manageable for IBL baking.
    // 1024 or 2048 is more than enough for high-quality IBL.
    if (processed.GetMetadata().width > MAX_INPUT_SIZE || processed.GetMetadata().height > MAX_INPUT_SIZE ||
        (!isCubemap && processed.GetMetadata().width != processed.GetMetadata().height))
    {
        int targetSize = (std::min)((int)MAX_INPUT_SIZE, (std::max)((int)processed.GetMetadata().width, (int)processed.GetMetadata().height));
        printf("Input %s is large or non-square (%zux%zu). Resizing to %dx%d for baking...\n", 
               isCubemap ? "cubemap" : "texture", processed.GetMetadata().width, processed.GetMetadata().height, targetSize, targetSize);
        ScratchImage resized;
        if (SUCCEEDED(Resize(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), 
                             targetSize, targetSize, TEX_FILTER_CUBIC, resized)))
        {
            processed = std::move(resized);
        }
    }

    // MANDATORY: Generate a full mip chain for the source texture if it's missing or after resize.
    // Filtered importance sampling and irradiance baking require mips for smoothness.
    if (processed.GetMetadata().mipLevels <= 1)
    {
        printf("Generating mipmaps for source %s...\n", isCubemap ? "cubemap" : "texture");
        ScratchImage mips;
        if (SUCCEEDED(GenerateMipMaps(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), 
                                     TEX_FILTER_DEFAULT, 0, mips)))
        {
            processed = std::move(mips);
        }
        else
        {
            printf("Error: Failed to generate mipmaps for %s.\n", isCubemap ? "cubemap" : "texture");
            assert(false && "Failed to generate mipmaps");
            return false;
        }
    }

    cm.size = (int)processed.GetMetadata().width;
    cm.mipCount = (int)processed.GetMetadata().mipLevels;

    int numFaces = isCubemap ? CUBEMAP_FACES : 1;
    int totalFloats = 0;
    int s = cm.size;
    for (int m = 0; m < cm.mipCount; ++m)
    {
        totalFloats += s * s * numFaces * COMPONENTS_PER_PIXEL;
        s = std::max(1, s / 2);
    }
    cm.data.resize(totalFloats);

    int floatOffset = 0;
    for (int m = 0; m < cm.mipCount; ++m)
    {
        int mipSize = std::max(1, cm.size >> m);
        for (int face = 0; face < numFaces; ++face)
        {
            const Image* img = processed.GetImage(m, face, 0);
            if (!img) 
            {
                printf("Error: Failed to get image for mip %d, face %d\n", m, face);
                return false;
            }
            memcpy(&cm.data[floatOffset], img->pixels, mipSize * mipSize * BYTES_PER_PIXEL);
            floatOffset += mipSize * mipSize * COMPONENTS_PER_PIXEL;
        }
    }

    return true;
}

void SaveDDS(std::string_view path, const Cubemap& cm, bool isCubemap)
{
    std::filesystem::path p(path);
    std::wstring wpath = p.wstring();
    
    TexMetadata metadata = {};
    metadata.width = cm.size;
    metadata.height = cm.size;
    metadata.depth = 1;
    metadata.arraySize = isCubemap ? CUBEMAP_FACES : 1;
    metadata.mipLevels = cm.mipCount;
    metadata.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    metadata.dimension = TEX_DIMENSION_TEXTURE2D;
    metadata.miscFlags = isCubemap ? TEX_MISC_TEXTURECUBE : 0;

    ScratchImage image;
    if (FAILED(image.Initialize(metadata)))
    {
        printf("Error: Failed to initialize ScratchImage for saving.\n");
        assert(false && "Failed to initialize ScratchImage for saving");
        return;
    }
    int numFaces = isCubemap ? 6 : 1;
    int floatOffset = 0;
    for (int m = 0; m < cm.mipCount; ++m)
    {
        int mipSize = std::max(1, cm.size >> m);
        for (int face = 0; face < numFaces; ++face)
        {
            const Image* img = image.GetImage(m, face, 0);
            memcpy(img->pixels, &cm.data[floatOffset], mipSize * mipSize * BYTES_PER_PIXEL);
            floatOffset += mipSize * mipSize * COMPONENTS_PER_PIXEL;
        }
    }

    ScratchImage compressed;
    if (SUCCEEDED(Compress(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DXGI_FORMAT_BC6H_UF16, TEX_COMPRESS_PARALLEL, TEX_THRESHOLD_DEFAULT, compressed)))
    {
        SaveToDDSFile(compressed.GetImages(), compressed.GetImageCount(), compressed.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
    }
    else
    {
        // std::cerr << "Warning: Failed to compress to BC6H_UF16. Saving as R32G32B32A32_FLOAT." << std::endl;
        // SaveToDDSFile(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
        printf("Error: Failed to compress to BC6H_UF16.\n");
        assert(false && "Failed to compress to BC6H_UF16");
    }
}

int main(int argc, char** argv)
{
    std::filesystem::path inputFile;
    int irrSize = DEFAULT_IRR_SIZE;
    int radSize = DEFAULT_RAD_SIZE;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            showHelp = true;
            break;
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            irrSize = std::stoi(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc)
        {
            radSize = std::stoi(argv[++i]);
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

    //inputFile = "D:\\Workspace\\GLTF Scenes\\RTXPT-Assets\\EnvironmentMaps\\simplebluesky.dds";

    if (showHelp || inputFile.empty())
    {
        printf("Usage: DDSToIBL <input.dds> [-i irrSize] [-r radSize]\n");
        printf("Convert DDS cubemap or 2D texture to IBL irradiance and radiance maps.\n");
        printf("Supported formats: All DDS formats including BC6H (via DirectXTex).\n");
        printf("Baking is performed on CUDA.\n");
        printf("\n");
        printf("Options:\n");
        printf("  -i irrSize       Irradiance map size (default 64 for cubemaps, input size for 2D)\n");
        printf("  -r radSize       Radiance map size (default 256 for cubemaps, input size for 2D)\n");
        printf("  --help           Show this help message\n");
        return showHelp ? 0 : 1;
    }

    if (irrSize <= 0) irrSize = DEFAULT_IRR_SIZE;
    if (radSize <= 0) radSize = DEFAULT_RAD_SIZE;

    Cubemap env;
    bool isCubemap;
    if (!LoadDDSTexture(inputFile.string(), env, isCubemap))
    {
        printf("Failed to load %s\n", inputFile.string().c_str());
        return 1;
    }

    printf("Loaded %s: size %d, mips %d\n", isCubemap ? "cubemap" : "texture", env.size, env.mipCount);

    int numFaces = isCubemap ? 6 : 1;

    Cubemap irradiance;
    irradiance.size = irrSize;
    irradiance.mipCount = 1;
    irradiance.data.resize(irradiance.size * irradiance.size * numFaces * COMPONENTS_PER_PIXEL);

    Cubemap radiance;
    radiance.size = radSize;
    radiance.mipCount = 0;
    int s = radSize;
    while (s > 0)
    {
        radiance.mipCount++;
        s /= 2;
    }
    
    int totalFloats = 0;
    s = radSize;
    for (int m = 0; m < radiance.mipCount; ++m)
    {
        totalFloats += s * s * numFaces * COMPONENTS_PER_PIXEL;
        s /= 2;
        if (s == 0) s = 1;
    }
    radiance.data.resize(totalFloats);

    printf("Baking Irradiance...\n");
    BakeIrradianceCUDA(env, irradiance, numFaces);
    printf("Baking Radiance...\n");
    BakeRadianceCUDA(env, radiance, numFaces);

    // Generate output filenames
    std::filesystem::path baseFilename = inputFile;
    baseFilename.replace_extension("");

    std::filesystem::path irrOutput = baseFilename;
    irrOutput += "_irradiance.dds";
    std::filesystem::path radOutput = baseFilename;
    radOutput += "_radiance.dds";

    printf("Saving %s...\n", irrOutput.string().c_str());
    SaveDDS(irrOutput.string(), irradiance, isCubemap);
    printf("Saving %s...\n", radOutput.string().c_str());
    SaveDDS(radOutput.string(), radiance, isCubemap);

    printf("Done.\n");

    return 0;
}
