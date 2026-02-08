# DDS to IBL

A small, minimal C++ application that generates radiance and irradiance DDS cubemaps from input linear HDR DDS cubemaps, using CUDA.

## Usage

```
DDSToIBL <input.dds> [-i irrSize] [-r radSize]
```

### Inputs
- **input.dds**: A linear HDR DDS cubemap or 2D texture in one of the supported formats

### Outputs
- **<input>_irradiance.dds**: Irradiance cubemap or 2D texture (compressed to BC6H_UF16)
- **<input>_radiance.dds**: Radiance cubemap or 2D texture with full mip chain (compressed to BC6H_UF16)

### Options
- `-i irrSize`: Irradiance map size (default 64 for cubemaps, input size for 2D textures)
- `-r radSize`: Radiance map size (default 256 for cubemaps, input size for 2D textures)
- `--help`: Show help message

## Dependencies

- [DirectXTex](https://github.com/microsoft/DirectXTex) (as a submodule)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)

## Supported Input Formats

- R32G32B32A32_FLOAT
- R16G16B16A16_FLOAT
- R11G11B10_FLOAT
- BC6H_UF16
- BC6H_SF16