# 固定选模规则与结论

{
  "status": "COMPLETE_NO_PILOT_GAIN",
  "selected": "SV00_G1",
  "passing_candidates": [],
  "flags": {
    "SV02_SAMV_GEOM": {
      "nondecrease": {
        "replica_probe2": false,
        "cf_probe2": true
      },
      "D2_crossing": false,
      "target_met": false,
      "material_met": false
    },
    "SV04_SAMV_FC": {
      "nondecrease": {
        "replica_probe2": false,
        "cf_probe2": true
      },
      "D2_crossing": true,
      "target_met": false,
      "material_met": false
    },
    "SV05_COMBINED": {
      "nondecrease": {
        "replica_probe2": false,
        "cf_probe2": true
      },
      "D2_crossing": true,
      "target_met": false,
      "material_met": false
    }
  },
  "target_met": false,
  "material_met": false,
  "deployment": "N0_UNCHANGED",
  "independent_confirmation": false,
  "automatic_expansion": false,
  "identity": "5dddcf586f194d0e7f6f8a444030d2bd78d2df5526604d5c99246d470867af15"
}

所有判定使用未舍入 fraction：每个候选五项指标在两个池均不低于G1−1e−10；CF APall>D2+1e−10且AP50≥D2−1e−10；额外 MATERIAL 需 APall−D2≥.001。按 CF APall、CF AP50、Replica APall、CF mIoU 排序，1e−10内视作相等，再优先SV02、SV04、SV05。

最佳简单对照为 SV01_SAM2_GEOM，按同一固定词典序比较，仅作对照报告。

本轮未达到固定晋升条件；保持 G1 研究参照，不开展自动全量扩展。
