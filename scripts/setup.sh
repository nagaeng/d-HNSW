#!/bin/bash
#
# System setup for DHNSW nodes
#
# This script:
#   1. Formats and mounts a data disk at /users/feifang/dhnsw
#   2. Installs Mellanox OFED driver
#   3. Installs build tools and system libraries
#   4. Configures huge pages for RDMA
#
# Supports Ubuntu 18.04 and 20.04.
# A reboot is required after running this script.
#

set -e

# ============================================
# Install build tools and system libraries
# ============================================
echo "Installing system packages..."
sudo apt update -y
sudo apt install -y \
    build-essential autoconf libtool pkg-config \
    g++ cmake clang gdb tmux \
    python3-pip numactl sysstat zstd \
    libtbb-dev libgtest-dev libboost-all-dev \
    google-perftools libgoogle-perftools-dev \
    libssl-dev libgflags-dev libnuma-dev \
    libblas-dev libopenblas-dev libatlas-base-dev \
    libjemalloc-dev \
    protobuf-compiler libprotobuf-dev libc-ares-dev \
    software-properties-common \
    linux-tools-common linux-tools-$(uname -r) 2>/dev/null || true

# ============================================
# Install Mellanox OFED
# ============================================
ubuntu_version=$(lsb_release -r -s)
echo "Detected Ubuntu $ubuntu_version"

if [ "$ubuntu_version" == "18.04" ]; then
    OFED_URL="https://content.mellanox.com/ofed/MLNX_OFED-4.9-5.1.0.0/MLNX_OFED_LINUX-4.9-5.1.0.0-ubuntu18.04-x86_64.tgz"
elif [ "$ubuntu_version" == "20.04" ]; then
    OFED_URL="https://content.mellanox.com/ofed/MLNX_OFED-4.9-5.1.0.0/MLNX_OFED_LINUX-4.9-5.1.0.0-ubuntu20.04-x86_64.tgz"
else
    echo "Warning: Unsupported Ubuntu version $ubuntu_version for OFED, skipping OFED install."
    OFED_URL=""
fi

if [ -n "$OFED_URL" ]; then
    echo "Installing Mellanox OFED..."
    mkdir -p /tmp/ofed_install && cd /tmp/ofed_install
    if [ ! -f ofed.tgz ]; then
        wget -O ofed.tgz "$OFED_URL"
    fi
    tar zxf ofed.tgz
    cd MLNX*
    sudo ./mlnxofedinstall --force
    sudo /etc/init.d/openibd restart
    sudo /etc/init.d/opensmd restart || true
    cd /
    rm -rf /tmp/ofed_install
fi

# ============================================
# Install gtest
# ============================================
if [ -d "/usr/src/gtest" ]; then
    echo "Building gtest..."
    cd /usr/src/gtest
    sudo cmake .
    sudo make
    sudo cp /usr/src/gtest/lib/libgtest*.a /usr/local/lib/ 2>/dev/null || \
    sudo cp /usr/src/gtest/libgtest*.a /usr/local/lib/ 2>/dev/null || true
    sudo cp -r /usr/src/gtest/include/gtest /usr/local/include/ 2>/dev/null || true
fi

# ============================================
# Configure huge pages for RDMA
# ============================================
echo "Configuring huge pages..."
sudo sh -c 'echo 1400 > /proc/sys/vm/nr_hugepages'

# Make persistent across reboots
if ! grep -q "vm.nr_hugepages" /etc/sysctl.conf; then
    echo "vm.nr_hugepages = 1400" | sudo tee -a /etc/sysctl.conf
fi

cat /proc/meminfo | grep Huge

# ============================================
# Set stack size
# ============================================
ulimit -s 16384

echo ""
echo "============================================"
echo "System setup completed."
echo "A reboot is required for OFED to take effect."
echo "============================================"
