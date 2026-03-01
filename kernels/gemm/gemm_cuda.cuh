#pragma once

void launch_sgemm_naive(const float* A, const float* B, float* C,
                         int M, int N, int K);
void launch_sgemm_tiled(const float* A, const float* B, float* C,
                         int M, int N, int K);
void launch_sgemm_reg_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K);
