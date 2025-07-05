set -e

if [ -d build ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

mkdir build
cd build
cmake ..
make -j$(nproc)
echo "Build completed successfully."
