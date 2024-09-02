# Install CUDA on Ubuntu 20.04 
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-tk-and-driver)

## Requierement

### Verify You Have a CUDA-Capable GPU

To verify that your GPU is CUDA-capable, go to your distribution’s equivalent of System Properties, or, from the command line, enter:
```
lspci | grep -i nvidia
```
If you do not see any settings, update the PCI hardware database that Linux maintains by entering update-pciids (generally found in /sbin) at the command line and rerun the previous lspci command.
If your graphics card is from NVIDIA and it is listed in [here](https://developer.nvidia.com/cuda-gpus), your GPU is CUDA-capable.

### Verify You Have a Supported Version of Linux
To determine which distribution and release number you’re running, type the following at the command line:
```
uname -m && cat /etc/*release
```
### Verify the System Has gcc Installed
The gcc compiler is required for development using the CUDA Toolkit. 
To verify the version of gcc installed on your system, type the following on the command line:
```
gcc --version
```
If an error message displays, you need to install the development tools from your Linux distribution or obtain a version of gcc and its accompanying toolchain from the Web.

### Verify the System has the Correct Kernel Headers and Development Packages Installed
The CUDA Driver requires that the kernel headers and development packages for the running version of the kernel be installed at the time of the driver installation, as well whenever the driver is rebuilt. For example, if your system is running kernel version 3.17.4-301, the 3.17.4-301 kernel headers and development packages must also be installed.

While the Runfile installation performs no package validation, the RPM and Deb installations of the driver will make an attempt to install the kernel header and development packages if no version of these packages is currently installed. However, it will install the latest version of these packages, which may or may not match the version of the kernel your system is using. Therefore, it is best to manually ensure the correct version of the kernel headers and development packages are installed prior to installing the CUDA Drivers, as well as whenever you change the kernel version.

The version of the kernel your system is running can be found by running the following command:
```
uname -r
```
This is the version of the kernel headers and development packages that must be installed prior to installing the CUDA Drivers. This command will be used multiple times below to specify the version of the packages to install. Note that below are the common-case scenarios for kernel usage. More advanced cases, such as custom kernel branches, should ensure that their kernel headers and sources match the kernel build they are running.

Download cuda toolkit for DL2 Lab machine Ubuntu 20.04 : [CUDA Toolkit 12.3 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)

- Operating system: Linux
- Architecutre: x86_64
- Distribution: Ubuntu
- Version: 20.04

With Installer type deb (network) :
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install #CUDA*
We picked here cuda-12-3 
```
[CUDA*](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages)

# Download cuDNN

Cuda is a library that allows you to use the GPU efficiently. However, to use your GPU even more efficiently, cuDNN implements some standard operations for Deep Neural Networks such as forward propagation, backpropagation for convolutions, pooling, normalization, etc. In order to use Pytorch and Tensorflow, you need to install cuDNN.

- [Download cuDNN](https://developer.nvidia.com/cudnn)
- [Compatibility Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)

Find the version corresponding to your version of CUDA and download it.
Extract all folders:
- lib
- include
- bin
```
Copy <cudnn_path>\bin\cudnn\*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin
Copy <cudnn_path>\cuda\include\cudnn\*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\include
Copy <cudnn_path>\cuda\lib\x64\cudnn\*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\lib\x64
```
You can then delete cuDNN folder

# Download Pytorch

Follow the instruction on [Pytorch](https://pytorch.org/get-started/locally/)

# Install accelerate

In your environment:
```
pip install accelerate
```
or
```
conda install -c conda-forge accelerate
```

# Install Bitsandbytes
Bitsandbytes requires accelerate

In your environment
You first need to install the Hugging Face Library Transformer :
```
pip install transformers
```
Install Bitsandbytes
```
pip install transformers bitsandbytes>=0.39.0 -q
```
