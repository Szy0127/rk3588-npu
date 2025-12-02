#!/bin/bash
# Quick start script: build and deploy in one command

set -e

echo "=========================================="
echo "RK3588-NPU Android Build & Deploy"
echo "=========================================="
echo ""

# Step 1: Build
echo "[1/2] Building for Android..."
./build-android.sh

# Step 2: Deploy
echo ""
echo "[2/2] Deploying to Android device..."
./deploy-android.sh

echo ""
echo "=========================================="
echo "Done! Library is ready on your device."
echo "=========================================="

