originally using distribution 19.10 drivers
===========================================

https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-19-10-eoan-ermine-linux

```
ubuntu-drivers devices
sudo apt install nvidia-driver-435
```

Use PPA?
========

-	https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa

```shell
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

After adding PPA

```shell
❯ sudo apt list --upgradable
Listing... Done
google-chrome-stable/stable 80.0.3987.106-1 amd64 [upgradable from: 80.0.3987.100-1]
libvdpau-dev/eoan 1.3-0ubuntu0~gpu19.10.1 amd64 [upgradable from: 1.2-1ubuntu1]
libvdpau1/eoan 1.3-0ubuntu0~gpu19.10.1 amd64 [upgradable from: 1.2-1ubuntu1]
libvulkan-dev/eoan 1.1.126.0-2~gpu19.10.1 amd64 [upgradable from: 1.1.114.0-1]
libvulkan1/eoan 1.1.126.0-2~gpu19.10.1 amd64 [upgradable from: 1.1.114.0-1]
libxnvctrl0/eoan 440.44-0ubuntu0.19.10.1 amd64 [upgradable from: 435.21-0ubuntu2]
nvidia-settings/eoan 440.44-0ubuntu0.19.10.1 amd64 [upgradable from: 435.21-0ubuntu2]
vdpau-driver-all/eoan 1.3-0ubuntu0~gpu19.10.1 amd64 [upgradable from: 1.2-1ubuntu1]


❯ ubuntu-drivers devices

== /sys/devices/pci0000:00/0000:00:02.0/0000:02:00.0 ==
modalias : pci:v000010DEd00001E84sv00003842sd00003175bc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-440 - third-party free recommended
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-430 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

install newer version (`440`) (`435` is current install)

```shell
sudo apt install nvidia-settings nvidia-driver-440
```
