# My-Torch-Extension
![avatar](./image/Logo.png)

This is a minimalist and extensible PyTorch extension for implementing custom backend operators in PyTorch.We provide a simple addition operator, demonstrating how to expose C++ code to Python for invocation using [pybind11](https://pybind11.readthedocs.io/en/stable/index.html). Additionally, we showcase how to package this project using [setuptools](https://setuptools.pypa.io/en/latest/).Based on this project, you can more quickly compare the performance and functionality of custom CUDA kernels with PyTorch's built-in kernels.


## Quick Start

First, you need to have an NVIDIA GPU and have CUDA installed correctly. Then, you can install the Python environment required for this project using the following command: 
```bash
pip install -r requirements.txt
```
> Of course, if you have already installed the required environment, you can skip this step.


### Build && Install
Build & Install using the following command is recommended:
```bash
python3 setup.py develop
```
You can also check your installation using the following command:
```bash
pip list
```
### Run the example

```bash
python3 examples/example_add.py
```


## How to extend operators
First, you need to implement your CUDA kernel in a file within the `csrc`directory, for example, `csrc/attention.cu`. Then, you need to declare your operator invocation interface in the `csrc/ops.h` file. After that, you also need to register this interface to the ` pybind11::module ops` in the `csrc/pybind.cpp` file. Once done, you can rebuild and reinstall the package.

At this point, you can use the following example code to utilize your custom operator:

```python
import torch
import my_torch_ext 

# Call the module
my_torch_ext.ops.attention()
```

## TODO
* Support for more GPU backends such as HIP and OpenCL,etc.
* ...

## Reference
* [vLLM](https://github.com/vllm-project/vllm)
* [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)






