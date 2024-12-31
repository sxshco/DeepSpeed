# Installation

The quickest way to get started with DeepSpeed is via pip, this will install
the latest release of DeepSpeed which is not tied to specific PyTorch or CUDA
versions. DeepSpeed includes several C++/CUDA extensions that we commonly refer
to as our 'ops'.  By default, all of these extensions/ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html) to build and
dynamically link them at runtime.

## Requirements
* [PyTorch](https://pytorch.org/) must be installed _before_ installing DeepSpeed.
* For full feature support we recommend a version of PyTorch that is >= 1.9 and ideally the latest PyTorch stable release.
* A CUDA or ROCm compiler such as [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction) or [hipcc](https://github.com/ROCm-Developer-Tools/HIPCC) used to compile C++/CUDA/HIP extensions.
* Specific GPUs we develop and test against are listed below, this doesn't mean your GPU will not work if it doesn't fall into this category it's just DeepSpeed is most well tested on the following:
  * NVIDIA: Pascal, Volta, Ampere, and Hopper architectures
  * AMD: MI100 and MI200

## PyPI
We regularly push releases to [PyPI](https://pypi.org/project/deepspeed/) and encourage users to install from there in most cases.

```bash
pip install deepspeed
```

After installation, you can validate your install and see which extensions/ops
your machine is compatible with via the DeepSpeed environment report.

```bash
ds_report
```

If you would like to pre-install any of the DeepSpeed extensions/ops (instead
of JIT compiling) or install pre-compiled ops via PyPI please see our [advanced
installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/).

## Windows
Windows support is partially supported with DeepSpeed. On Windows you can build wheel with following steps, currently only inference mode is supported.
1. Install pytorch, such as pytorch 1.8 + cuda 11.1
2. Install visual cpp build tools, such as VS2019 C++ x64/x86 build tools
3. Launch cmd console with Administrator privilege for creating required symlink folders
4. Run `python setup.py bdist_wheel` to build wheel in `dist` folder
