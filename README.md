# DDS to IBL

A C++ command-line tool that generates Image-Based Lighting (IBL) assets from DDS textures using CUDA acceleration. Converts HDR DDS cubemaps or equirectangular 2D textures into irradiance and radiance cubemaps, and optionally bakes BRDF Look-Up Tables (LUTs) for physically-based rendering.

## Usage

```
DDSToIBL <input.dds> [options]
```

### Examples

```bash
# Basic usage - convert equirectangular to IBL cubemaps
DDSToIBL environment.hdr

# Custom sizes and samples
DDSToIBL input.dds -i 128 -r 512 -s 2048 -m 4096

# Bake only BRDF LUT
DDSToIBL --brdf-only

# Bake BRDF LUT with custom parameters
DDSToIBL -b -l 512 -n 2048
```

### Inputs
- **input.dds**: A linear HDR DDS cubemap or equirectangular 2D texture in any supported DDS format
- If input is equirectangular, it's automatically converted to a cubemap

### Outputs
- **<input>_irradiance.dds**: Irradiance cubemap (diffuse IBL) - compressed to BC6H_UF16
- **<input>_radiance.dds**: Radiance cubemap with full mip chain (specular IBL) - compressed to BC6H_UF16
- **brdf_lut.dds**: BRDF LUT (when `-b` or `--brdf-only` is used) - saved as R16G16_FLOAT

### Options
- `-h, --help`: Show help message
- `-i <size>`: Irradiance cubemap face size (default: 64)
- `-r <size>`: Radiance cubemap face size (default: 256)
- `-s <count>`: Number of samples for irradiance baking (default: 1024)
- `-m <count>`: Number of samples for radiance baking (default: 8192)
- `-b, --brdf`: Bake BRDF LUT alongside IBL cubemaps
- `-l <size>`: BRDF LUT size (default: 256)
- `-n <count>`: Number of samples for BRDF LUT baking (default: 1024)
- `--brdf-only`: Bake only BRDF LUT (ignores input texture)

## Baking Details

### Irradiance Baking
- Uses Monte Carlo integration over hemisphere
- Default: 1024 samples per texel

### Radiance Baking
- Prefiltered environment mapping using GGX importance sampling
- Roughness increases with mip level (0 to 1)
- Default: 8192 samples per texel
- Full mip chain generated for efficient specular lookups

### BRDF LUT Baking
- Cook-Torrance BRDF integration for split-sum approximation
- Stores scale and bias factors for Fresnel and geometry terms
- Default: 1024 samples per texel

### Prerequisites
- CUDA Toolkit (with NVIDIA GPU)

## Dependencies

- **[CUDA](https://developer.nvidia.com/cuda-toolkit)**: GPU acceleration for baking computations
