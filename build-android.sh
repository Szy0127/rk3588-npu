#!/bin/bash
# Build script for Android using NDK r28-beta2

set -e

NDK_PATH="/home/szy/android-ndk-r28-beta2"
CROSS_FILE="android-cross-file.txt"
BUILD_DIR="build-android"

# Check if NDK exists
if [ ! -d "$NDK_PATH" ]; then
    echo "Error: NDK not found at $NDK_PATH"
    exit 1
fi

# Check if cross file exists
if [ ! -f "$CROSS_FILE" ]; then
    echo "Error: Cross file not found: $CROSS_FILE"
    exit 1
fi

echo "Cleaning previous build..."
rm -rf "$BUILD_DIR"

echo "Configuring Meson build for Android..."
meson setup "$BUILD_DIR" \
    --cross-file "$CROSS_FILE" \
    --buildtype release

echo "Building..."
cd "$BUILD_DIR"
ninja

echo ""
echo "Build completed successfully!"
echo "Output files:"
find . -name "*.so" -o -name "*.a" | grep -v "^\./meson" | head -10

