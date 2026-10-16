# Handoff

科学终态：COMPLETE_A_NO_RESIDUAL_SIGNAL。B/C严格未激活；部署N0保持。无需补原缓存：279份原预测NPZ与原文本完整，A读取真实向量而无GPU新推理。

`--parent-root`在本轮唯一表示STRUCT_ROOT；与上轮表示FG_ROOT的参数语义不同。FG/LR/PR/ATTR从STRUCT的已验binding解析，父JSON/缓存只读。已有同任务可恢复，不调用父all，不扫描全历史。

本地继续：
```bash
python scripts/evaluation/run_semantic_update_evidence_v1.py --parent-root "$STRUCT_ROOT" --output-root "$OUT" --phase all --resume
```

控制环境继承父ovimap-map Python；BLAS1。A不加载FC/SAM；GPU若指定必须等于继承GPU0，不能擅换卡。公开主数据：A/source_scores.npz为完整原FP32分数、defined、valid_ids；A/source_vectors.npz为9个主列clean8V的真实768向量和原文本，original_column_indices按全局columns连接。A/import.json含全部key/类/列、源SHA与类别分割；A/predictions.npz含110候选与OOF预测。可由replay.py重建A输入并复算；不存在从200分数伪逆恢复向量。

B/C阶段入口在当前闭合gate下统一返回NOT_ACTIVATED_BY_A；B模型定义/算子为未训练代码，不声明已验证B/C服务器接入。若未来科学合同要改变门槛、输入或数学，必须新identity并重新审计，不能沿用本轮负结果收据宣称完成新实验。

没有外部全方法复现、主分支合并或部署变更。C1固定成果，C2仅C1收据与永久链接。C2 SHA及真实零工作resume结果写OUT/publication，不产生第三次自引用提交。
