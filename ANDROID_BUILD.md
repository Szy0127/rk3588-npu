# Android 交叉编译指南

本指南说明如何在 x86 Linux 主机上使用 Android NDK 交叉编译 rk3588-npu 库，并部署到 Android 设备。

## 前置要求

1. **Android NDK r28-beta2** 已安装在 `/home/szy/android-ndk-r28-beta2`
2. **Meson** 构建系统（可通过 pip 安装：`pip install meson`）
3. **Ninja** 构建工具（通常随 Meson 一起安装）
4. **ADB**（Android Debug Bridge）用于部署到设备

## 编译步骤

### 1. 使用构建脚本（推荐）

```bash
./build-android.sh
```

这将：
- 清理之前的构建
- 配置 Meson 使用 Android 交叉编译工具链
- 编译库文件

### 2. 手动编译

```bash
# 清理之前的构建
rm -rf build-android

# 配置构建
meson setup build-android \
    --cross-file android-cross-file.txt \
    --buildtype release

# 编译
cd build-android
ninja
```

编译完成后，库文件位于 `build-android/librk3588-npu.so`

## 部署到 Android 设备

### 使用部署脚本（推荐）

1. 确保 Android 设备已连接并启用 USB 调试：
   ```bash
   adb devices
   ```

2. 运行部署脚本：
   ```bash
   ./deploy-android.sh
   ```

脚本会自动：
- 检查设备连接
- 在设备上创建目录 `/data/local/tmp/rk3588-npu`
- 复制库文件和头文件到设备

### 手动部署

```bash
# 在设备上创建目录
adb shell "mkdir -p /data/local/tmp/rk3588-npu"

# 复制库文件
adb push build-android/librk3588-npu.so /data/local/tmp/rk3588-npu/

# 复制头文件（可选，用于开发）
adb shell "mkdir -p /data/local/tmp/rk3588-npu/include"
adb push include/*.h /data/local/tmp/rk3588-npu/include/
```

## 在 Android 应用中使用

### 方法 1: 使用 Android.mk (NDK Build)

在 `jni/Android.mk` 中添加：

```makefile
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := rk3588-npu
LOCAL_SRC_FILES := ../libs/arm64-v8a/librk3588-npu.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := your-module
LOCAL_SRC_FILES := your-code.c
LOCAL_SHARED_LIBRARIES := rk3588-npu
include $(BUILD_SHARED_LIBRARY)
```

### 方法 2: 使用 CMakeLists.txt

```cmake
add_library(rk3588-npu SHARED IMPORTED)
set_target_properties(rk3588-npu PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/librk3588-npu.so
)

target_link_libraries(your-target rk3588-npu)
```

### 方法 3: 直接复制到应用

将 `librk3588-npu.so` 复制到：
```
app/src/main/jniLibs/arm64-v8a/librk3588-npu.so
```

然后在 Java/Kotlin 代码中加载：
```java
static {
    System.loadLibrary("rk3588-npu");
}
```

## 检查依赖

编译后的库主要依赖 Android 系统库，可以通过以下命令检查：

```bash
readelf -d build-android/librk3588-npu.so | grep NEEDED
```

通常只需要：
- `libc.so` (Android bionic C library)
- `libm.so` (数学库，如果使用了数学函数)
- `liblog.so` (Android 日志库)

这些库在 Android 系统中都已提供，无需额外部署。

## 注意事项

1. **API 级别**: 当前配置使用 Android API 21 (Android 5.0+)。如需支持更低版本，修改 `android-cross-file.txt` 中的 `aarch64-linux-android21` 为对应版本。

2. **设备权限**: 访问 `/dev/dri/card1` 需要 root 权限或设备特定的 SELinux 策略。

3. **内核支持**: 确保 Android 设备的内核支持 RKNPU DRM 驱动。

4. **架构匹配**: 当前配置为 aarch64 (arm64-v8a)。如果设备是 32 位 ARM，需要修改交叉编译配置。

## 故障排除

### 编译错误：找不到头文件
- 检查 NDK 路径是否正确
- 确认 sysroot 路径存在

### 运行时错误：库加载失败
- 确认库文件架构与设备匹配（arm64-v8a）
- 检查库文件权限：`adb shell chmod 755 /path/to/librk3588-npu.so`

### 设备连接问题
- 运行 `adb devices` 检查设备是否被识别
- 确认 USB 调试已启用
- 尝试 `adb kill-server && adb start-server`

## 文件说明

- `android-cross-file.txt`: Meson 交叉编译配置文件
- `build-android.sh`: 自动化构建脚本
- `deploy-android.sh`: 自动化部署脚本
- `include/drm-compat.h`: Android 兼容的 DRM 头文件（自动使用，无需手动包含）

