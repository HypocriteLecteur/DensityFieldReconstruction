## Environment Setup
1. Create python virtual environment with: \
```conda create -n mv-dfr Python=3.12``` \
```conda activate mv-dfr```
2. Install torch with: \
```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128```
3. Install packages with: \
```pip install -r environment.txt```
4. Install density field rasterizer(s) with: \
```cd density_field_rasterizer/gaussian_rasterizer_simple_small``` \
```pip install --no-build-isolation .```
Repeat for the other variants as needed (replace `_small` with `_large` or `_small_decoupled`).

### Troubleshooting rasterizer installation

**Non-English Windows: UnicodeDecodeError in cpp_extension**

On Windows with a non-English system locale (e.g., Chinese cp936), `torch.utils.cpp_extension` may fail with `UnicodeDecodeError: 'cp1' codec can't decode bytes`. The OEM codec cannot decode MSVC compiler output that contains Unicode characters (e.g., CUDA headers with special characters).

Fix: edit `<conda_env>/Lib/site-packages/torch/utils/cpp_extension.py` line 46, changing:
```python
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
```
to:
```python
SUBPROCESS_DECODE_ARGS = ('oem', 'replace') if IS_WINDOWS else ()
```

**CUDA 13.3+: CCCL preprocessor errors**

CUDA 13.3 ships CCCL/CUB headers that require the MSVC standard-conforming preprocessor. Using the traditional preprocessor produces errors like `expected a "{"` or `"detail" is ambiguous` in `cub/block/*.cuh`.

The setup.py already includes `-Xcompiler /Zc:preprocessor` to enable conforming mode. If building on an older CUDA toolkit that does not support this flag, replace it with:
```python
'-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
```

**DLL load failed at import time**

The rasterizer `.pyd` depends on PyTorch DLLs (`c10.dll`, `torch_cpu.dll`, `torch_python.dll`). Always `import torch` **before** importing any rasterizer module so the DLLs are already loaded into the process.

## Run Code
1. All codes should run in root directory, meaning the output of ```os.getcwd()``` should be the root directory.
2. 