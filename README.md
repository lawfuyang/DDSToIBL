# DDS to IBL

A C++ command-line tool that generates Image-Based Lighting (IBL) assets from DDS textures using CUDA acceleration. Converts HDR DDS cubemaps or equirectangular 2D textures into irradiance and radiance cubemaps optimized for physically-based rendering.

### Dependencies

- **[CUDA Runtime](https://developer.nvidia.com/cuda-toolkit)**: GPU acceleration for baking computations

## Usage

```
DDSToIBL <input.dds/.hdr> [options]
```

### Examples

```bash
# Basic usage - convert equirectangular to IBL cubemaps
DDSToIBL environment.hdr

# Custom sizes and samples for higher quality
DDSToIBL input.dds -i 128 -r 512 -s 2048 -m 4096

# Bake BRDF LUT alongside IBL cubemaps
DDSToIBL input.dds -b

# Bake only BRDF LUT with custom parameters
DDSToIBL --brdf-only -l 512 -n 2048
```

### Inputs

- **input.dds/.hdr**: Linear HDR texture file
  - DDS cubemaps in any supported format
  - Equirectangular 2D textures (.hdr format supported)
  - If equirectangular, automatically converted to cubemap

### Outputs

- **<input>_irradiance.dds**: Irradiance cubemap (diffuse IBL)
  - Face size: configurable (default: 64x64)
  - Format: BC6H_UF16 (compressed HDR)
  - Used for diffuse environment lighting

- **<input>_radiance.dds**: Radiance cubemap with full mip chain (specular IBL)
  - Base face size: configurable (default: 256x256)
  - Format: BC6H_UF16 (compressed HDR)
  - Mip levels: roughness increases with mip level (0 to 1)
  - Used for specular environment reflections

- **brdf_lut.dds**: BRDF Look-Up Table (when `-b` or `--brdf-only` is used)
  - Size: configurable (default: 256x256)
  - Format: R16G16_FLOAT
  - Stores precomputed scale/bias factors for Fresnel and geometry terms

### Options

- `-h, --help`: Show help message
- `-i <size>`: Irradiance cubemap face size (default: 64)
- `-r <size>`: Radiance cubemap base face size (default: 256)
- `-s <count>`: Number of samples for irradiance baking (default: 1024)
- `-m <count>`: Number of samples for radiance baking (default: 8192)
- `-b, --brdf`: Bake BRDF LUT alongside IBL cubemaps
- `-l <size>`: BRDF LUT size (default: 256)
- `-n <count>`: Number of samples for BRDF LUT baking (default: 1024)
- `--brdf-only`: Bake only BRDF LUT (ignores input texture)

## Technical Details

### Irradiance Baking
- Uses Monte Carlo integration over hemisphere
- Samples environment lighting for diffuse reflection
- Default: 1024 samples per texel for quality/performance balance

### Radiance Baking
- Prefiltered environment mapping using GGX importance sampling
- Roughness increases linearly with mip level (0.0 at base to 1.0 at finest mip)
- Generates full mip chain for efficient specular lookups at varying roughness
- Default: 8192 samples per texel for high-quality specular reflections

### BRDF LUT Baking
- Cook-Torrance BRDF integration for split-sum approximation
- Stores scale (red) and bias (green) factors for Fresnel and geometry terms
- Enables real-time evaluation of specular BRDF during rendering
- Default: 1024 samples per texel
