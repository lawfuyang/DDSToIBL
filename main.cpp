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

bool LoadDDSTexture(std::string_view path, TextureData& td)
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

    // Performance Optimization: If the input is extremely large (e.g. 4K), 
    // downsample it to something manageable for IBL baking.
    if (processed.GetMetadata().width > MAX_INPUT_SIZE || processed.GetMetadata().height > MAX_INPUT_SIZE)
    {
        float aspect = (float)processed.GetMetadata().width / (float)processed.GetMetadata().height;
        size_t targetW, targetH;
        if (aspect >= 1.0f) {
            targetW = MAX_INPUT_SIZE;
            targetH = (size_t)((float)MAX_INPUT_SIZE / aspect);
        } else {
            targetH = MAX_INPUT_SIZE;
            targetW = (size_t)((float)MAX_INPUT_SIZE * aspect);
        }
        
        printf("Input is large (%zux%zu). Resizing to %zux%zu for baking...\n", 
               processed.GetMetadata().width, processed.GetMetadata().height, targetW, targetH);
               
        ScratchImage resized;
        if (SUCCEEDED(Resize(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), 
                             targetW, targetH, TEX_FILTER_CUBIC, resized)))
        {
            processed = std::move(resized);
        }
    }

    // MANDATORY: Generate a full mip chain for the source texture if it's missing or after resize.
    if (processed.GetMetadata().mipLevels <= 1)
    {
        printf("Generating mipmaps for source %s...\n", isCubemap ? "cubemap" : "texture");
        ScratchImage mips;
        if (SUCCEEDED(GenerateMipMaps(processed.GetImages(), processed.GetImageCount(), processed.GetMetadata(), 
                                     TEX_FILTER_DEFAULT, 0, mips)))
        {
            processed = std::move(mips);
        }
    }

    td.width = (int)processed.GetMetadata().width;
    td.height = (int)processed.GetMetadata().height;
    td.mipCount = (int)processed.GetMetadata().mipLevels;
    td.numFaces = isCubemap ? CUBEMAP_FACES : 1;

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
            memcpy(&td.data[floatOffset], img->pixels, (size_t)mipW * mipH * BYTES_PER_PIXEL);
            floatOffset += (size_t)mipW * mipH * COMPONENTS_PER_PIXEL;
        }
    }

    return true;
}

void SaveDDS(std::string_view path, const TextureData& td)
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

    ScratchImage compressed;
    if (SUCCEEDED(Compress(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DXGI_FORMAT_BC6H_UF16, TEX_COMPRESS_PARALLEL, TEX_THRESHOLD_DEFAULT, compressed)))
    {
        SaveToDDSFile(compressed.GetImages(), compressed.GetImageCount(), compressed.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
    }
    else
    {
        printf("Warning: Failed to compress to BC6H_UF16. Saving as R32G32B32A32_FLOAT.\n");
        SaveToDDSFile(image.GetImages(), image.GetImageCount(), image.GetMetadata(), DDS_FLAGS_NONE, wpath.c_str());
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

    TextureData env;
    if (!LoadDDSTexture(inputFile.string(), env))
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

    printf("Baking Irradiance...\n");
    BakeIrradianceCUDA(env, irradiance);
    printf("Baking Radiance...\n");
    BakeRadianceCUDA(env, radiance);

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

    printf("Done.\n");

    return 0;
}
