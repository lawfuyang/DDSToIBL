# DDS to IBL

A small, minimal C++ application that generates radiance and irradiance DDS cubemaps from input linear HDR DDS cubemaps or equirectangular 2D textures (converted to cubemaps), using CUDA.

## Usage

```
DDSToIBL <input.dds> [-i irrSize] [-r radSize]
```

### Inputs
- **input.dds**: A linear HDR DDS cubemap or equirectangular 2D texture in one of the supported formats. If the input is an equirectangular 2D texture, it will be converted to a cubemap as output.

### Outputs
- **<input>_irradiance.dds**: Irradiance cubemap (compressed to BC6H_UF16)
- **<input>_radiance.dds**: Radiance cubemap with full mip chain (compressed to BC6H_UF16)

### Options
- `-i irrSize`: Irradiance map size (default 64 for cubemaps, input size for 2D textures)
- `-r radSize`: Radiance map size (default 256 for cubemaps, input size for 2D textures)
- `--help`: Show help message

## Baking Details

- Radiance baking: ~1000 samples per texel
- Irradiance baking: 8196 samples per texel

## Dependencies

- [DirectXTex](https://github.com/microsoft/DirectXTex) (as a submodule)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)

## Supported Input Formats

- R32G32B32A32_FLOAT
- R16G16B16A16_FLOAT
- R11G11B10_FLOAT
- BC6H_UF16
- BC6H_SF16