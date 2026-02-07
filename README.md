# DDS to IBL

A small, minimal C++ application that generates radiance and irradiance DDS cubemaps from input linear HDR DDS cubemaps, using CUDA.

This is 100% ghetto, hardcoded, AI slop, but it suits your needs if it's all you need.

## Dependencies

- DirectXTex (as a submodule)

## Supported Input Formats

- R32G32B32A32_FLOAT
- R16G16B16A16_FLOAT
- R11G11B10_FLOAT
- BC6H_UF16
- BC6H_SF16