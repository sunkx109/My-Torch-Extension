# My-Torch-Extension
![avatar](./image/Logo.png)

这是一个极简的、可拓展的Pytorch Extension，用于给Pytorch自定义后端算子. 我们提供了一个简单的加法算子，展示了如何通过[pybind11](https://pybind11.readthedocs.io/en/stable/index.html)将C++代码暴露接口给Python调用，同时也展示了如何通过[setuptools](https://setuptools.pypa.io/en/latest/)对本project进行封装. 基于本project您可以更快地对比自定义的CUDA kernel与Pytorch自身的kernel的性能及功能.


## Quick Start

首先，你需要有一个NVIDIA GPU 并且已经正确安装了CUDA，然后您就能通过如下命令安装本项目所需要的Python环境 :
```bash
pip install -r requirements.txt
```
> 当然如果您已经正确安装了所需要环境您可以直接跳过这一步

### Build && Install

推荐使用如下命令 Build && Install
```bash
python3 setup.py develop
```
您也可以通过如下命令检查您的安装
```bash
pip list
```

### Run the example
```bash
python3 examples/example_add.py
```

## How to extend operators
首先，你需要在`csrc`目录下实现你的cuda kernel，比如`csrc/attention.cu`, 然后你需要在`csrc/ops.h`文件中声明你的算子调用接口，然后你还需要在`csrc/pybind.cpp` 文件中将这个接口注册到` pybind11::module ops`. 之后，您便可以重新Build && Install.

至此，您就可以通过以下的示例代码来使用您的自定义算子 :

```python
import torch
import my_torch_ext 

# Call the module
my_torch_ext.ops.attention()
```

## TODO
* 支持更多的GPU后端比如 : HIP 、OpenCL,etc.
* ...

## Reference
* [vLLM](https://github.com/vllm-project/vllm)
* [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)