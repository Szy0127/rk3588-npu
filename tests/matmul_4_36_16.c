/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <sys/mman.h>

#ifdef __ANDROID__
#include "drm-compat.h"
#else
#include <libdrm/drm.h>
#endif

#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"

  // Test currently runs against kernel 5.10 haven't tested 6.1 kernel.

  // Test data is from test-mul-mat.cpp see https://github.com/ggerganov/ggml
  // matrix A (4 X 36)
  float matrixA[] = {
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    };

  // matrix B (16 X 36)
  float matrixB[] = {
    9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
    4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
    8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
    5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
    9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
    9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
    2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
    8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
    8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
    8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
    10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
    };

  // matrix C (4 x 16)
  float expected_result[] = {
    1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
    1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
    1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
    999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
    };

  uint64_t npu_regs[112];

int main(int argc, char **argv) {

  int M=4;
  int K=36;
  int N=16;
  unsigned int core_id = 0;  // Default to core 0
  int ret=0;

  if (argc == 2) {
    core_id = atoi(argv[1]);
    if (core_id > 3) {
      printf("Invalid core_id %d, must be 0-3\n", core_id);
      return -1;
    }
  } else if (argc > 2) {
    printf("Invalid number of args %d\n", argc);
    printf("Usage: %s [core_id]\n", argv[0]);
    printf("  core_id: optional, 0-3 (default: 0)\n");
    return -1;
  }

  // Open DRI called "rknpu"
  int fd = npu_open();

  uint64_t regcmd_dma, regcmd_obj;
  uint32_t regcmd_handle;
  uint64_t *regcmd = mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);

  uint64_t tasks_dma, tasks_obj;
  uint32_t tasks_handle;
  struct rknpu_task *tasks = mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);

  uint64_t input_dma, input_obj;
  uint32_t input_handle;
  void *input = mem_allocate(fd, 4096, &input_dma, &input_obj, 0, &input_handle);

  uint64_t weights_dma, weights_obj;
  uint32_t weights_handle;
  void *weights = mem_allocate(fd, 4096, &weights_dma, &weights_obj, 0, &weights_handle);

  uint64_t output_dma, output_obj;
  uint32_t output_handle;
  void *output = mem_allocate(fd, 4096, &output_dma, &output_obj, 0, &output_handle);

  printf("input dma is %lx, output dma is %lx, weights dma is %lx\n", input_dma, output_dma, weights_dma);
  if ((regcmd == NULL) || (tasks == NULL) || (input == NULL) || (weights == NULL) || (output == NULL)) {
    printf("Failed to allocate memory \n");
    exit(1);
  }

  // Reset the NPU
  npu_reset(fd);

  matmul_params_t params;
  params.m = M;
  params.k = 64;
  params.n = N;
  params.input_dma = input_dma;
  params.weights_dma = weights_dma;
  params.output_dma = output_dma;
  params.tasks = (uint64_t *) &npu_regs;
  params.fp32tofp16 = 0;
  ret = gen_matmul_fp16(&params);
  if (ret !=0) {
    printf("gen_matmul_fp16 failed %d\n",ret);
    goto cleanup;
  }

  memcpy((void *)regcmd,(void *)&npu_regs,sizeof(npu_regs));

  memset((void *)input,0,M*64*sizeof(__fp16));
  memset((void *)weights,0,64*N*sizeof(__fp16));
  memset((void *)output,0,M*N*sizeof(float));

  tasks[0].flags  = 0;
  tasks[0].op_idx = 0;
  tasks[0].enable_mask = 0xd;
  tasks[0].int_mask = 0x300; // wait for DPU to finish
  tasks[0].int_clear = 0x1ffff;
  tasks[0].int_status = 0;
  tasks[0].regcfg_amount = (sizeof(npu_regs)/sizeof(uint64_t))-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
  tasks[0].regcfg_offset = 0;
  tasks[0].regcmd_addr = regcmd_dma;

  __fp16 *weights_fp16 = weights;

  for(int n=1;n<=N;n++) {
    for(int k=1;k<=K;k++) {
      weights_fp16[weight_fp16(64,n,k)]= matrixB[((n-1)*K)+(k-1)];
    }
  }

  __fp16 *feature_data_fp16 = (__fp16*) input;

  for (int m=1;m<=M;m++) {
    for (int k=1;k<=K;k++) {
      feature_data_fp16[feature_data(64,4,1,8,k,m,1)]= matrixA[((m-1)*K)+(k-1)];
    }
  }

  // Initialize subcore_task array
  struct rknpu_subcore_task subcore_tasks[5] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
  subcore_tasks[core_id].task_start = 0;
  subcore_tasks[core_id].task_number = 1;
  
  printf("M is %d, K is %d, N is %d, using NPU core %d\n", M, K, N, core_id);
  
  struct rknpu_submit submit = {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = tasks_obj,
    .regcfg_obj_addr = 0,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = 1 << core_id,  // Set bit for the specified core
    .fence_fd = -1,
    .subcore_task = {
      subcore_tasks[0],
      subcore_tasks[1],
      subcore_tasks[2],
      subcore_tasks[3],
      subcore_tasks[4]
    },
  };
  ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
  printf("RKNPU_SUBMIT returned %d\n", ret);
  if (ret <0) {
    return ret;
  }

  printf("=========================================================================================================\n");
  printf("Comparing CPU (expected) vs NPU (actual) results...\n");
  float *output_data = (float*) output;
  int total_elements = M * N;
  int matched = 0;
  int mismatched = 0;
  int max_mismatches_to_print = 10;  // Limit printed mismatches
  const float abs_tolerance = 0.1f;  // Absolute tolerance for float comparison
  const float rel_tolerance = 0.001f;  // Relative tolerance (0.1%)
  
  printf("\nNPU Output Matrix:\n");
  for (int m=1;m<=M;m++) {
    for (int n=1;n<=N;n++) {
      float actual = output_data[feature_data(N, 4, 1, 4, n, m, 1)];
      printf("%6.1f ", actual);
    }
    printf("\n");
  }
  printf("\n");
  
  for (int m=1;m<=M;m++) {
    for (int n=1;n<=N;n++) {
      float actual = output_data[feature_data(N, 4, 1, 4, n, m, 1)];
      float expected = expected_result[((m-1)*N)+(n-1)];
      
      // Use relative and absolute tolerance for float comparison
      float diff = fabsf(actual - expected);
      float max_val = fmaxf(fabsf(actual), fabsf(expected));
      float rel_error = max_val > 0.0f ? diff / max_val : diff;
      
      if (diff <= abs_tolerance || rel_error <= rel_tolerance) {
        matched++;
      } else {
        mismatched++;
        if (mismatched <= max_mismatches_to_print) {
          printf("MISMATCH [m:%d, n:%d]  CPU(expected):%10.1f  NPU(actual):%10.1f  diff:%10.6f  rel_err:%.4f%%\n",
                 m, n, expected, actual, diff, rel_error * 100.0f);
        }
        ret = -1;
      }
    }
  }
  
  printf("---------------------------------------------------------------------------------------------------------\n");
  printf("Comparison Summary:\n");
  printf("  Total elements: %d\n", total_elements);
  printf("  Matched: %d (%.2f%%)\n", matched, (matched * 100.0f) / total_elements);
  printf("  Mismatched: %d (%.2f%%)\n", mismatched, (mismatched * 100.0f) / total_elements);
  printf("  Tolerance: abs=%.6f, rel=%.4f%%\n", abs_tolerance, rel_tolerance * 100.0f);
  
  if (ret == 0) {
    printf("✓ Multiplication of [%d,%d] x [%d,%d] successful - All results match!\n", M, K, N, K);
  } else {
    printf("✗ Multiplication of [%d,%d] x [%d,%d] FAILED - Found %d mismatches\n", M, K, N, K, mismatched);
    if (mismatched > max_mismatches_to_print) {
      printf("  (Only first %d mismatches shown)\n", max_mismatches_to_print);
    }
  }
  printf("=========================================================================================================\n");

cleanup:
  munmap(regcmd,1024);
  munmap(tasks,1024);
  munmap(input,4096);
  munmap(weights,4096);
  munmap(output,4096);

  mem_destroy(fd, regcmd_handle, regcmd_obj);
  mem_destroy(fd, tasks_handle, tasks_obj );
  mem_destroy(fd, input_handle, input_obj);
  mem_destroy(fd, weights_handle, weights_obj);
  mem_destroy(fd, output_handle, output_obj);

  npu_close(fd);
  return ret;
}
