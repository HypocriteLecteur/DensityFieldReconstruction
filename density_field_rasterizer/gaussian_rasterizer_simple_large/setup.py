from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import platform

# Get DEBUG environment variable
DEBUG_MODE = False

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Define common arguments
glm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'glm')
print(f"GLM include path: {glm_path}")

common_nvcc_args = [
    f'-I{glm_path}',
    '--default-stream', 'per-thread',
    '-lineinfo'
]

common_cxx_args = []
common_link_args = []

if platform.system() == "Windows":
    common_cxx_args.append(f'/I{glm_path}') 
else: # Linux/macOS
    common_cxx_args.append(f'-I{glm_path}')

# Add debug flags if DEBUG_MODE is enabled
if DEBUG_MODE:
    print("Building in DEBUG mode...")
    # NVCC flags for debug: -G for debug info, -O0 for no optimization
    common_nvcc_args.extend(['-G', '-O0'])
    # CXX flags for debug on Windows: /Zi for PDB, /Od for no optimization
    if platform.system() == "Windows":
        common_cxx_args.extend(['/Zi', '/Od'])
        common_link_args.append('/DEBUG') # Linker flag for PDB
    else: # Linux/macOS: -g for debug info, -O0 for no optimization
        common_cxx_args.extend(['-g', '-O0'])
        # For Linux, no extra linker flag usually needed for debug symbols
else:
    print("Building in RELEASE mode...")
    common_nvcc_args.append('-O3')
    common_nvcc_args.append('--fmad=true')
    common_nvcc_args.append('--use_fast_math')
    if platform.system() == "Windows":
        common_cxx_args.append('-O2')
    else:
        common_cxx_args.append('-O2')


setup(
    name="gaussian_rasterizer_simple_large",
    ext_modules=[
        CUDAExtension(
            name="gaussian_rasterizer_simple_large",
            sources=[
                'ext.cpp',
                'forward.cu',
                'backward.cu'
            ],
            extra_compile_args={
                'nvcc': common_nvcc_args,
                'cxx': common_cxx_args
            },
            extra_link_args=common_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)