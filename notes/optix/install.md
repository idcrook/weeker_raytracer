
# Install

- Ubuntu Linux x64 19.10

Downloaded SDKs from Nvidia developer site.

```
# needs to be writeable by user
mkdir -p /usr/local/nvidia/

sh NVIDIA-OptiX-SDK-6.5.0-linux64.sh --skip-license \
  --prefix=/usr/local/nvidia --include-subdir

sh NVIDIA-OptiX-SDK-7.0.0-linux64.sh --skip-license \
   --prefix=/usr/local/nvidia --include-subdir
```
