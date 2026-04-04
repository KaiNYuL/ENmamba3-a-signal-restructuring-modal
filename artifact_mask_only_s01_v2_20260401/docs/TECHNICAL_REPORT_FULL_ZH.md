# Fusion 主模型技术报告（命名修正版）

## 0. Fusion 架构与参数配置（文档开头总览）

### 0.1 架构原理

主模型为双分支时序重建结构：

1. Bi-Mamba3 分支提取高阶状态空间表示。
2. Bi-Mamba 分支提取稳定状态空间表示。
3. 在 token 级进行门控融合，得到最终重建表征。

融合公式：

$$
H_{fusion} = g \cdot H_{mamba3} + (1-g) \cdot H_{mamba}
$$

$$
g = \sigma\left(W[H_{mamba3};H_{mamba}] + b\right)
$$

输出头采用 patch 重建线性层，将 token 表征还原为多通道时序信号。

### 0.2 主要参数配置（m3m-fusion-mask-restructruing）

本轮主模型采用 mask-only 重建，核心参数如下：

| 参数 | 取值 |
|---|---:|
| prediction_mode | encoder_mask_only |
| ssm_variant | fusion |
| d_model | 32 |
| d_state | 64 |
| headdim | 16 |
| n_bi_layers | 2 |
| patch_size | 96 |
| chunk_size | 32 |
| dropout | 0.15 |
| disable_preconv | true |
| preconv_kernel | 5 |
| encoder_random_mask_ratio | 0.15 |
| encoder_eval_mask_ratio | 0.15 |
| mask_observed_residual | true |
| lr | 1e-4 |
| weight_decay | 0.01 |
| batch_size | 8 |
| epochs | 120 |
| patience | 12 |

其中 `encoder_eval_mask_ratio` 固定为 0.15 以与训练掩码比例一致，避免评估阶段退化为无掩码重建。

## 1. 报告范围与本次修正

本报告基于 Fusion 主模型重建结果，采用 clean24（28 被试过滤后再剔除 4 个异常值）作为核心统计集合。

## 2. 数据与评估口径

- filtered 被试：28
- 异常值剔除：s15, s18, s30, s32
- clean 集合：24
- 指标口径：标准化空间 MSE / MAE / R2

文件：

- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_clean24_metrics.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_outliers_iqr.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_clean_summary_iqr.json

## 3. Baseline 对比（Fusion 为唯一主模型）

说明：baseline 区域仅包含 Fusion 与外部基线模型（深度/经典），不包含 mamba3。

表格：

- ../results/baselines_all/filtered_r2_ge0/all_baselines_comparison_table_fusion_primary_clean24.csv

核心均值（clean24）：

| model | n_subjects | mse_mean | mae_mean | r2_mean |
|---|---:|---:|---:|---:|
| fusion_primary_clean24 | 24 | 0.185862 | 0.113841 | 0.841564 |
| tcn_ae | 24 | 136909.849604 | 2.031063 | 0.517947 |
| ridge | 24 | 136903.216844 | 2.414709 | 0.141708 |
| timesnet_ae | 24 | 136918.362876 | 2.226615 | 0.093907 |
| pls | 24 | 136918.078642 | 2.271070 | 0.072014 |
| patch_transformer_ae | 24 | 147583.707846 | 4.250145 | 0.062920 |
| masked_transformer_ae | 24 | 147661.623497 | 4.263997 | 0.054804 |
| random_forest | 24 | 136920.226563 | 2.223584 | 0.046078 |

图像：

![all_models_metric](../results/baselines_all/filtered_r2_ge0/figures/all_models_metric_comparison_fusion_primary_clean24.png)

![r2_gap_vs_fusion](../results/baselines_all/filtered_r2_ge0/figures/all_models_r2_gap_vs_fusion_primary_clean24.png)

## 4. 消融实验（四路）

本节统一以 Fusion 为基准，比较 3 个消融对象：

1. mamba3
2. mamba
3. no_aux_bias

结果文件：

- ../results/ablation/no_aux_bias_filtered_r2_ge0/ablation_compare_fusion_primary_vs_mamba3_vs_mamba_vs_no_aux_bias_clean24.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/ablation_compare_fusion_primary_vs_mamba3_vs_mamba_vs_no_aux_bias_clean24_summary.json

关键均值（clean24）：

- Fusion R2 mean = 0.841564
- mamba3 R2 mean = 0.719090
- mamba R2 mean = 0.805093
- no_aux_bias R2 mean = 0.528771

相对 Fusion 的 R2 差值均值：

- mamba3 - fusion = -0.122474
- mamba - fusion = -0.036471
- no_aux_bias - fusion = -0.312793

图像：

![ablation_grouped](../results/ablation/no_aux_bias_filtered_r2_ge0/figures/r2_grouped_fusion_mamba3_mamba_no_aux_bias_clean24.png)

![ablation_delta](../results/ablation/no_aux_bias_filtered_r2_ge0/figures/delta_r2_mamba3_mamba_no_aux_minus_fusion_clean24.png)

消融结论：

1. mamba3 相比 Fusion 有明显退化。
2. mamba 相比 Fusion 有中等退化，但优于 mamba3。
3. no_aux_bias 退化最大，说明情绪评分偏置对当前任务有显著正向作用。

## 5. 通道级结果（Fusion, clean24）

文件：

- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_per_channel_test_mse.csv
- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_per_channel_test_summary.csv
- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_channel_mapping.json

关键统计：

1. 最优通道：ch_16（原始 #40），mse_mean = 0.1440
2. 最差通道：ch_13（原始 #37），mse_mean = 0.2586

图像：

![channel_bar](../results/summary/channel_test_fusion_primary_clean24/figures/fusion_primary_clean24_channel_test_mse_mean_std.png)

![channel_heatmap](../results/summary/channel_test_fusion_primary_clean24/figures/fusion_primary_clean24_channel_test_mse_subject_heatmap.png)

## 6. 最终结论

1. 在 clean24 上，Fusion 是当前最优主模型。
4. no_aux_bias 进一步验证了情绪评分偏置模块的必要性。

## 7. 资产索引

- ../results/summary/fusion_primary_clean24_report_assets.json

