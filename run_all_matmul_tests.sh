#!/data/data/com.termux/files/usr/bin/bash

# 批量测试 4 个矩阵乘法程序：
#   - matmul_int8
#   - matmul_fp16
#   - matmul_fp16_fp16
#   - matmul_4_36_16
#
# 要求：
#   - 多组参数
#   - 每组多次重复
#   - 统计每个程序的成功 / 失败次数
#   - 详细输出写入日志文件，终端只显示汇总

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# NPU core id（0~3），可通过第一个参数指定，默认为 0
CORE_ID="${1:-0}"
if ! [[ "$CORE_ID" =~ ^[0-3]$ ]]; then
  echo "Invalid CORE_ID: $CORE_ID (must be 0-3)"
  exit 1
fi

# 每组参数重复次数（可通过环境变量覆盖）
REPEAT_PER_CASE="${REPEAT_PER_CASE:-3}"

echo "Using NPU core: $CORE_ID"
echo "Repeat per case: $REPEAT_PER_CASE"
echo "Logs dir: $LOG_DIR"
echo "-------------------------------------------------------------------------------"

###############################################################################
# 公共的 (M, K, N) 参数集合
# 满足：
#   - 对 matmul_int8:  M<=544, K<=4096(32倍数), N<=4096(16倍数)
#   - 对 matmul_fp16:  M<=384, K<=4096(32倍数), N<=4096(16倍数)
#   - 对 matmul_fp16_fp16: M<=384, K<=8192(32倍数), N<=8192(16倍数)
###############################################################################
CASES_COMMON=(
  # 小尺寸
  #"4   32    16"
  "8   64    32"
  "16  128   64"
  "32  128   64"

  # 中等尺寸
  "32  256   128"
  "64  256   128"
  "64  512   256"
  "96  512   256"

  # 大一点
  "128 512   256"
  "128 1024  512"
  #"192 1024  512"

  # 更大（接近上限但留有余量）
  #"256 2048  1024"
  #"320 2048  1024"
  #"384 2048  1024"
)

run_prog_with_cases() {
  local prog="$1"
  shift
  local -n cases_ref="$1"

  local log_file="$LOG_DIR/${prog}_test_$(date +%Y%m%d_%H%M%S)_core${CORE_ID}.log"
  local total=0
  local ok=0
  local fail=0

  echo "=== Program: $prog (core_id=$CORE_ID) ==="
  echo "Log: $log_file"

  echo "Start tests for $prog (core_id=$CORE_ID, repeat=$REPEAT_PER_CASE)" >> "$log_file"

  for case in "${cases_ref[@]}"; do
    # shellcheck disable=SC2086
    set -- $case
    local M=$1
    local K=$2
    local N=$3

    echo "Case: M=$M K=$K N=$N (repeat $REPEAT_PER_CASE times)" | tee -a "$log_file"

    for ((i=1; i<=REPEAT_PER_CASE; i++)); do
      total=$((total+1))

      echo "  Run #$i: ./$prog $M $K $N $CORE_ID" >> "$log_file"

      ./"$prog" "$M" "$K" "$N" "$CORE_ID" >>"$log_file" 2>&1
      local ret=$?

      if [ $ret -eq 0 ]; then
        ok=$((ok+1))
        echo "    -> PASS (ret=$ret)" >> "$log_file"
      else
        fail=$((fail+1))
        echo "    -> FAIL (ret=$ret)" >> "$log_file"
      fi
    done

    echo "" >> "$log_file"
  done

  echo "Summary for $prog:"
  echo "  Total runs : $total"
  echo "  Passed     : $ok"
  echo "  Failed     : $fail"
  echo "  Log file   : $log_file"
  echo "-------------------------------------------------------------------------------"
}

###############################################################################
# 1) matmul_int8
###############################################################################
if [ -x "./matmul_int8" ]; then
  run_prog_with_cases "matmul_int8" CASES_COMMON
else
  echo "matmul_int8 not found or not executable, skip."
fi

###############################################################################
# 2) matmul_fp16
###############################################################################
if [ -x "./matmul_fp16" ]; then
  run_prog_with_cases "matmul_fp16" CASES_COMMON
else
  echo "matmul_fp16 not found or not executable, skip."
fi

###############################################################################
# 3) matmul_fp16_fp16
###############################################################################
if [ -x "./matmul_fp16_fp16" ]; then
  run_prog_with_cases "matmul_fp16_fp16" CASES_COMMON
else
  echo "matmul_fp16_fp16 not found or not executable, skip."
fi

###############################################################################
# 4) matmul_4_36_16 - 固定 4x36x16，不需要 M K N 参数
###############################################################################
if [ -x "./matmul_4_36_16" ]; then
  local log_file_43616="$LOG_DIR/matmul_4_36_16_test_$(date +%Y%m%d_%H%M%S)_core${CORE_ID}.log"
  local total_43616=0
  local ok_43616=0
  local fail_43616=0

  echo "=== Program: matmul_4_36_16 (core_id=$CORE_ID) ==="
  echo "Log: $log_file_43616"

  echo "Start tests for matmul_4_36_16 (core_id=$CORE_ID, repeat=$REPEAT_PER_CASE)" >> "$log_file_43616"

  for ((i=1; i<=REPEAT_PER_CASE; i++)); do
    total_43616=$((total_43616+1))

    echo "  Run #$i: ./matmul_4_36_16 $CORE_ID" >> "$log_file_43616"
    ./matmul_4_36_16 "$CORE_ID" >>"$log_file_43616" 2>&1
    ret=$?

    if [ $ret -eq 0 ]; then
      ok_43616=$((ok_43616+1))
      echo "    -> PASS (ret=$ret)" >> "$log_file_43616"
    else
      fail_43616=$((fail_43616+1))
      echo "    -> FAIL (ret=$ret)" >> "$log_file_43616"
    fi
  done

  echo "Summary for matmul_4_36_16:"
  echo "  Total runs : $total_43616"
  echo "  Passed     : $ok_43616"
  echo "  Failed     : $fail_43616"
  echo "  Log file   : $log_file_43616"
  echo "-------------------------------------------------------------------------------"
else
  echo "matmul_4_36_16 not found or not executable, skip."
fi

exit 0


