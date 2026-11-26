# Baseline v2 Report — Protocol B (Monthly Rolling × 5-Fold OOC)

- **Spec ID:** `baseline_v2_monthly_rolling_ooc_v1` (`v2.0.0`)
- **Timezone / Cutoff:** `Europe/Moscow (UTC+3)`, strict `event_time_msk < YYYY-MM-01 00:00:00 MSK`
- **Primary Metrics:** `Lift@1%` (monthly budget `K = ceil(1% * N_month) ~ 1,003` calls/month) & `PR-AUC`, reported **per-month first** with 95% CI, then **Macro-Monthly** & **Pooled OOF**.
- **L1 Caveat:** `L1_rule_target_lag` assumes `label_delay = 0` (`m-1` label available before month `m` cutoff); treat as reference upper bound until bank label SLA is verified.

## Tuned CatBoost Config (Selected Strictly on Val)
- `learning_rate=0.03`, `depth=6`, `l2_leaf_reg=25.0`, `median_best_iter_val=432`

## Locked Test — Per-Month Metrics (Protocol B)
| protocol | stage | month | exp_id | n_features | lift_1pct_mean | lift_1pct_ci_low | lift_1pct_ci_high | average_precision_mean | average_precision_ci_low | average_precision_ci_high | roc_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_rolling_ooc | test | 2022-11 | L0_base_rate | 0 | 1.1525 | 0.4642 | 1.8863 | 0.0087 | 0.0081 | 0.0094 | 0.5042 |
| B_rolling_ooc | test | 2022-12 | L0_base_rate | 0 | 0.7899 | 0.2555 | 1.4938 | 0.0075 | 0.0068 | 0.0081 | 0.4908 |
| B_rolling_ooc | test | 2023-01 | L0_base_rate | 0 | 0.7402 | 0.1489 | 1.4399 | 0.0068 | 0.0062 | 0.0075 | 0.5047 |
| B_rolling_ooc | test | 2022-11 | L2_rule_trx_rfm | 6 | 3.8033 | 2.6021 | 5.0712 | 0.0181 | 0.0159 | 0.0217 | 0.6480 |
| B_rolling_ooc | test | 2022-12 | L2_rule_trx_rfm | 6 | 3.8179 | 2.5790 | 5.3058 | 0.0158 | 0.0138 | 0.0187 | 0.6404 |
| B_rolling_ooc | test | 2023-01 | L2_rule_trx_rfm | 6 | 5.9214 | 4.1893 | 7.5588 | 0.0203 | 0.0171 | 0.0255 | 0.7303 |
| B_rolling_ooc | test | 2022-11 | B_logreg_T | 8 | 5.4169 | 4.0443 | 6.9398 | 0.0241 | 0.0214 | 0.0281 | 0.7493 |
| B_rolling_ooc | test | 2022-12 | B_logreg_T | 8 | 4.4762 | 3.0723 | 6.0007 | 0.0205 | 0.0180 | 0.0240 | 0.7291 |
| B_rolling_ooc | test | 2023-01 | B_logreg_T | 8 | 4.7371 | 3.1650 | 6.4094 | 0.0181 | 0.0156 | 0.0221 | 0.7379 |
| B_rolling_ooc | test | 2022-11 | B_logreg_TGD | 14 | 7.3762 | 5.6160 | 9.0618 | 0.0309 | 0.0272 | 0.0360 | 0.7736 |
| B_rolling_ooc | test | 2022-12 | B_logreg_TGD | 14 | 8.6891 | 6.6527 | 10.8962 | 0.0297 | 0.0252 | 0.0367 | 0.7599 |
| B_rolling_ooc | test | 2023-01 | B_logreg_TGD | 14 | 5.4773 | 3.8150 | 7.0946 | 0.0213 | 0.0182 | 0.0262 | 0.7546 |
| B_rolling_ooc | test | 2022-11 | B_catboost_TGD_tuned | 14 | 9.0666 | 7.2699 | 11.0380 | 0.0397 | 0.0335 | 0.0486 | 0.7765 |
| B_rolling_ooc | test | 2022-12 | B_catboost_TGD_tuned | 14 | 10.5761 | 8.4869 | 12.6589 | 0.0391 | 0.0325 | 0.0502 | 0.7704 |
| B_rolling_ooc | test | 2023-01 | B_catboost_TGD_tuned | 14 | 6.4642 | 4.5750 | 8.0548 | 0.0247 | 0.0211 | 0.0299 | 0.7701 |
| B_rolling_ooc | test | 2022-11 | L1_rule_target_lag | 3 | 7.3762 | 5.6579 | 9.0848 | 0.0161 | 0.0137 | 0.0200 | 0.5705 |
| B_rolling_ooc | test | 2022-12 | L1_rule_target_lag | 3 | 6.0560 | 4.5537 | 7.9415 | 0.0149 | 0.0124 | 0.0191 | 0.5822 |
| B_rolling_ooc | test | 2023-01 | L1_rule_target_lag | 3 | 10.0664 | 7.8043 | 12.3817 | 0.0216 | 0.0174 | 0.0281 | 0.6284 |

## Macro-Monthly Summary (Protocol B vs A)
| protocol | stage | exp_id | n_features | lift_1pct_mean | lift_1pct_ci_low | lift_1pct_ci_high | average_precision_mean | average_precision_ci_low | average_precision_ci_high | roc_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_rolling_ooc | val | L0_base_rate | 0 | 0.8057 | 0.4202 | 1.2281 | 0.0093 | 0.0088 | 0.0098 | 0.4997 |
| B_rolling_ooc | val | L2_rule_trx_rfm | 6 | 2.7953 | 2.1212 | 3.6635 | 0.0186 | 0.0171 | 0.0208 | 0.6477 |
| B_rolling_ooc | val | B_logreg_T | 8 | 3.7984 | 2.9747 | 4.7539 | 0.0227 | 0.0209 | 0.0255 | 0.7257 |
| B_rolling_ooc | val | B_logreg_TGD | 14 | 6.0264 | 5.0410 | 7.0777 | 0.0314 | 0.0289 | 0.0356 | 0.7631 |
| B_rolling_ooc | val | B_catboost_TGD_tuned | 14 | 8.6764 | 7.5767 | 9.8095 | 0.0374 | 0.0341 | 0.0426 | 0.7709 |
| B_rolling_ooc | val | L1_rule_target_lag | 3 | 6.1332 | 4.9935 | 7.2803 | 0.0200 | 0.0172 | 0.0244 | 0.5898 |
| A_flat_ooc | val | L0_base_rate | 0 | 0.8057 | 0.4202 | 1.2281 | 0.0093 | 0.0087 | 0.0099 | 0.4985 |
| A_flat_ooc | val | L2_rule_trx_rfm | 6 | 2.7953 | 2.1033 | 3.6367 | 0.0186 | 0.0172 | 0.0209 | 0.6476 |
| A_flat_ooc | val | B_logreg_T | 8 | 3.7408 | 2.8681 | 4.6930 | 0.0227 | 0.0209 | 0.0256 | 0.7258 |
| A_flat_ooc | val | B_logreg_TGD | 14 | 6.0264 | 4.9739 | 7.0721 | 0.0316 | 0.0291 | 0.0360 | 0.7635 |
| A_flat_ooc | val | B_catboost_TGD_tuned | 14 | 8.5421 | 7.5146 | 9.7113 | 0.0369 | 0.0338 | 0.0421 | 0.7703 |
| A_flat_ooc | val | L1_rule_target_lag | 3 | 6.1332 | 4.9935 | 7.2803 | 0.0200 | 0.0172 | 0.0244 | 0.5898 |
| B_rolling_ooc | test | L0_base_rate | 0 | 0.8942 | 0.4771 | 1.3563 | 0.0077 | 0.0073 | 0.0081 | 0.4999 |
| B_rolling_ooc | test | L2_rule_trx_rfm | 6 | 4.5142 | 3.6650 | 5.3492 | 0.0181 | 0.0167 | 0.0206 | 0.6729 |
| B_rolling_ooc | test | B_logreg_T | 8 | 4.8767 | 3.9343 | 5.8660 | 0.0209 | 0.0194 | 0.0235 | 0.7388 |
| B_rolling_ooc | test | B_logreg_TGD | 14 | 7.1809 | 6.0590 | 8.2769 | 0.0273 | 0.0252 | 0.0308 | 0.7627 |
| B_rolling_ooc | test | B_catboost_TGD_tuned | 14 | 8.7023 | 7.5304 | 9.7467 | 0.0345 | 0.0314 | 0.0397 | 0.7723 |
| B_rolling_ooc | test | L1_rule_target_lag | 3 | 7.8329 | 6.6440 | 9.1162 | 0.0176 | 0.0154 | 0.0207 | 0.5937 |
| A_flat_ooc | test | L0_base_rate | 0 | 0.8942 | 0.4771 | 1.3563 | 0.0077 | 0.0073 | 0.0081 | 0.5021 |
| A_flat_ooc | test | L2_rule_trx_rfm | 6 | 4.4649 | 3.7036 | 5.3922 | 0.0181 | 0.0167 | 0.0206 | 0.6728 |
| A_flat_ooc | test | B_logreg_T | 8 | 4.7342 | 3.8250 | 5.7772 | 0.0208 | 0.0193 | 0.0234 | 0.7379 |
| A_flat_ooc | test | B_logreg_TGD | 14 | 7.0492 | 5.8937 | 8.0603 | 0.0273 | 0.0252 | 0.0308 | 0.7625 |
| A_flat_ooc | test | B_catboost_TGD_tuned | 14 | 8.4171 | 7.3510 | 9.5358 | 0.0346 | 0.0314 | 0.0398 | 0.7722 |
| A_flat_ooc | test | L1_rule_target_lag | 3 | 7.8329 | 6.6440 | 9.1162 | 0.0176 | 0.0154 | 0.0207 | 0.5937 |

## Val-Only Ablations (Source T+G/T+D, Tenure, Linear+Tree Blend)
| group | exp_id | n_features | lift_1pct_mean | lift_1pct_ci_low | lift_1pct_ci_high | average_precision_mean | average_precision_ci_low | average_precision_ci_high | roc_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_main | B_logreg_T | 8 | 3.7984 | 2.9747 | 4.7539 | 0.0227 | 0.0209 | 0.0255 | 0.7257 |
| reference_main | B_logreg_TGD | 14 | 6.0264 | 5.0410 | 7.0777 | 0.0314 | 0.0289 | 0.0356 | 0.7631 |
| reference_main | B_catboost_TGD_tuned | 14 | 8.9368 | 7.7654 | 10.0576 | 0.0374 | 0.0341 | 0.0427 | 0.7705 |
| source_ablation | Abl_logreg_TG | 11 | 4.1848 | 3.2446 | 5.1074 | 0.0231 | 0.0212 | 0.0260 | 0.7273 |
| source_ablation | Abl_logreg_TD | 11 | 6.0182 | 4.9930 | 7.0676 | 0.0312 | 0.0287 | 0.0351 | 0.7624 |
| source_ablation | Abl_catboost_TG | 11 | 5.6812 | 4.6707 | 6.8241 | 0.0270 | 0.0241 | 0.0311 | 0.7311 |
| source_ablation | Abl_catboost_TD | 11 | 8.8629 | 7.6731 | 9.9827 | 0.0379 | 0.0345 | 0.0432 | 0.7700 |
| tenure_ablation | Abl_logreg_TGD_no_tenure | 13 | 4.8506 | 3.9276 | 5.8439 | 0.0283 | 0.0263 | 0.0314 | 0.7559 |
| tenure_ablation | Abl_logreg_TGD_cap365 | 14 | 5.4755 | 4.5634 | 6.5410 | 0.0310 | 0.0286 | 0.0352 | 0.7648 |
| tenure_ablation | Abl_catboost_TGD_no_tenure | 13 | 7.5228 | 6.5059 | 8.8652 | 0.0354 | 0.0323 | 0.0403 | 0.7623 |
| tenure_ablation | Abl_catboost_TGD_cap365 | 14 | 8.0735 | 7.0543 | 9.3494 | 0.0373 | 0.0341 | 0.0422 | 0.7707 |
| ensemble_ablation | Abl_blend_prob_50_50 | 14 | 8.2626 | 7.0977 | 9.4213 | 0.0360 | 0.0330 | 0.0412 | 0.7700 |
| ensemble_ablation | Abl_blend_prob_70_30 | 14 | 7.6460 | 6.4781 | 8.7488 | 0.0344 | 0.0315 | 0.0393 | 0.7682 |
| ensemble_ablation | Abl_blend_prob_30_70 | 14 | 8.7230 | 7.6411 | 9.9548 | 0.0370 | 0.0337 | 0.0425 | 0.7708 |
| ensemble_ablation | Abl_blend_rank_50_50 | 14 | 7.6953 | 6.5923 | 8.8269 | 0.0357 | 0.0326 | 0.0411 | 0.7700 |
| ensemble_ablation | Abl_blend_logit_50_50 | 14 | 8.2626 | 7.0744 | 9.4560 | 0.0362 | 0.0331 | 0.0416 | 0.7700 |

