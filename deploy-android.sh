#!/bin/bash
# Deploy script to copy binaries and dependencies to Android device

set -e

BUILD_DIR="build-android"
DEVICE_PATH="/data/data/com.termux/files/home/rk3588-npu"
ADB="${ADB:-adb}"

# Check if adb is available
if ! command -v "$ADB" &> /dev/null; then
    echo "Error: adb not found. Please install Android SDK platform-tools"
    exit 1
fi

# Check if device is connected
if ! "$ADB" devices | grep -q "device$"; then
    echo "Error: No Android device connected or authorized"
    echo "Please connect your device and enable USB debugging"
    exit 1
fi

echo "Device connected:"
"$ADB" devices

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Please run ./build-android.sh first"
    exit 1
fi

# Find the library
LIB_FILE=$(find "$BUILD_DIR" -name "librk3588-npu.so" | head -1)
if [ -z "$LIB_FILE" ]; then
    echo "Error: librk3588-npu.so not found in $BUILD_DIR"
    exit 1
fi

# Find test executables
TEST_EXECUTABLES=$(find "$BUILD_DIR" -type f -executable -name "matmul_*" ! -name "*.so" ! -name "*.a" | grep -v "^\./meson" || true)

echo ""
echo "Found library: $LIB_FILE"
if [ -n "$TEST_EXECUTABLES" ]; then
    echo "Found test executables:"
    echo "$TEST_EXECUTABLES" | sed 's/^/  /'
else
    echo "No test executables found"
fi

# Create directory on device
echo ""
echo "Creating directory on device: $DEVICE_PATH"
"$ADB" shell "mkdir -p $DEVICE_PATH"
"$ADB" shell "chmod 755 $DEVICE_PATH"

# Copy library to device
echo "Copying library to device..."
"$ADB" push "$LIB_FILE" "$DEVICE_PATH/"

# Copy test executables to device
if [ -n "$TEST_EXECUTABLES" ]; then
    echo ""
    echo "Copying test executables to device..."
    echo "$TEST_EXECUTABLES" | while read -r exec_file; do
        if [ -f "$exec_file" ]; then
            exec_name=$(basename "$exec_file")
            echo "  Copying $exec_name..."
            "$ADB" push "$exec_file" "$DEVICE_PATH/"
            "$ADB" shell "chmod 755 $DEVICE_PATH/$exec_name"
        fi
    done
fi

# Copy header files (for development)
echo "Copying header files..."
"$ADB" shell "mkdir -p $DEVICE_PATH/include"
for header in include/*.h; do
    if [ -f "$header" ]; then
        "$ADB" push "$header" "$DEVICE_PATH/include/"
    fi
done

# Get library dependencies (if any)
echo ""
echo "Checking library dependencies..."
if command -v readelf &> /dev/null; then
    DEPS=$(readelf -d "$LIB_FILE" 2>/dev/null | grep "NEEDED" | sed 's/.*\[\(.*\)\]/\1/' || true)
    if [ -n "$DEPS" ]; then
        echo "Dependencies found:"
        echo "$DEPS"
        echo ""
        echo "Note: These are Android system libraries and should be available on the device."
    else
        echo "No external dependencies found (static or Android system libraries)"
    fi
else
    echo "readelf not found, skipping dependency check"
fi

echo ""
echo "Deployment completed!"
echo ""
echo "Library location on device: $DEVICE_PATH/librk3588-npu.so"
if [ -n "$TEST_EXECUTABLES" ]; then
    echo "Test executables location: $DEVICE_PATH/"
    echo "$TEST_EXECUTABLES" | while read -r exec_file; do
        exec_name=$(basename "$exec_file")
        echo "  - $DEVICE_PATH/$exec_name"
    done
    echo ""
    echo "To run a test on the device:"
    echo "  adb shell $DEVICE_PATH/matmul_fp16 1 32 16"
fi
echo "Header files location: $DEVICE_PATH/include/"
echo ""
echo "To use the library in your Android app, add to Android.mk or CMakeLists.txt:"
echo "  LOCAL_SHARED_LIBRARIES := rk3588-npu"
echo "  LOCAL_LDFLAGS := -L\$(LOCAL_PATH)/libs -lrk3588-npu"
echo ""
echo "Or copy to your app's jniLibs directory:"
echo "  app/src/main/jniLibs/arm64-v8a/librk3588-npu.so"

