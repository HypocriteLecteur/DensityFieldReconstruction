## Environment Setup
1. Create python virtual environment with: \
```conda create -n mv-dfr Python=3.12``` \
```conda activate mv-dfr```
2. Install torch with: \
```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128```
3. Install packages with: \
```pip install -r environment.txt```
4. Install density field rasterizer with: \
```cd density_field_rasterizer/gaussian_rasterizer_simple_small``` \
```python setup.py install```
try build if install fails

## Run Code
1. All codes should run in root directory, meaning the output of ```os.getcwd()``` should be the root directory.
2. 