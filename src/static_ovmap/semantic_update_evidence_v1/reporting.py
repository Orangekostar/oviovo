"""Compact actual A evidence; inactive scientific cells stay explicitly empty."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from .common import REPO,DOCS,ARTIFACTS,read,write,verified,digest_file,gzip_json,save_arrays
from .readers import column_index

def percent(x):return 'null' if x is None else f'{100*x:.2f}'

def partial_report(binding,science):
    root=Path(binding['output_root']);docs=REPO/DOCS;dest=REPO/ARTIFACTS
    docs.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    observed=verified(root/'execution_observed.json')
    if observed['actual_exit_code']==0:raise ValueError('PARTIAL science must have a nonzero observed exit')
    text='# '+science['status']+'\n\n'+science.get('error','Dependency or numerical stage could not complete')+'\n\n缺失/技术失败不构成负科学结论，主表为null；保留已完成的独立阶段，不重编码旧RGB，不虚构样本或缩减分母。实际科学exit='+str(observed['actual_exit_code'])+'。\n'
    for name in ('RESULTS.md','HANDOFF.md','CLAIMS_LEDGER.md','DATA_SCOPE.md','EXTERNAL_CONTROL_SCOPE.md'):
        (docs/name).write_text(text)
    for name in ('science.json','execution_observed.json'):shutil.copyfile(root/name,dest/name)
    manifest=write(dest/'artifact_index.json',dict(status='PARTIAL',files=[dict(path=n,sha256=digest_file(dest/n)) for n in ('science.json','execution_observed.json')]))
    return write(root/'report.json',dict(status=science['status'],scientific_actual_exit_code=observed['actual_exit_code'],artifact_index=manifest['identity']))

def report(binding):
    root=Path(binding['output_root']);science=verified(root/'science.json')
    if science['status'].startswith('PARTIAL_'):return partial_report(binding,science)
    summary=verified(root/'A/summary.json')
    observed=verified(root/'execution_observed.json');meta=verified(root/'A/import.json')
    docs=REPO/DOCS;dest=REPO/ARTIFACTS;docs.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    if science['status']!='COMPLETE_A_NO_RESIDUAL_SIGNAL':raise ValueError('Report implementation is scoped to the observed A-negative transaction')
    machine=['science.json','execution_observed.json','A/import.json','A/summary.json','A/decision.json','A/predictions.npz']
    machine+=['A/'+p.name for p in sorted((root/'A').glob('*.json.gz'))]
    for name in machine:
        source=root/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name in ('history/repair_plan.json','history/primary_repair_comparison.json','review/test_results.json','bootstrap/archive_and_sources.json','bootstrap/task_paths.json'):
        if (root/name).exists():
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    shutil.copyfile(root/'class_split.json',dest/'class_split.json')
    # Minimal real source bundle: all imported scores/defined plus ONLY the nine used clean8V vectors.
    primary=[column_index(meta['columns'],'B_FC',None)]+[column_index(meta['columns'],m,s) for m in ('S_LL','G_LL','V_LL','MV_ATTN') for s in (17,29)]
    with np.load(root/'A/import.npz',allow_pickle=False) as a:
        scores=a['scores'];defined=a['defined'];vectors=a['embeddings'][:,primary,0];ids=a['valid_ids'];text=a['text']
    save_arrays(dest/'A/source_scores.npz',full200_cosines=scores,defined=defined,valid_ids=ids)
    save_arrays(dest/'A/source_vectors.npz',clean8V_embeddings=vectors,original_column_indices=np.array(primary),valid_ids=ids,text_embeddings=text)
    with np.load(dest/'A/source_scores.npz',allow_pickle=False) as a:
        if a['full200_cosines'].tobytes()!=scores.tobytes() or not np.array_equal(a['defined'],defined):raise ValueError('Source score export not exact')
    with np.load(dest/'A/source_vectors.npz',allow_pickle=False) as a:
        if a['clean8V_embeddings'].tobytes()!=vectors.tobytes() or a['text_embeddings'].tobytes()!=text.tobytes():raise ValueError('Source vector export not exact')
    from .plotting import render
    render(root,dest/'figures')
    rows=summary['LOFO_rows']+summary['main_rows']
    table=['|方法|seed|B_macro %|N_macro %|H_macro %|H_micro %|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];table.append('|'+r['method']+'|'+('静态一次' if r['seed'] is None else str(r['seed']))+'|'+ '|'.join(percent(m[k]) for k in ('B_macro','N_macro','H_macro','H_micro'))+'|')
    comparisons=summary['decision']['comparisons'];gate_table=['|方法|seed|ΔH_macro vs LOFO (pp)|正增益家族/可定义家族|B门槛|','|---|---:|---:|---:|---|']
    for method in ('S_LL','G_LL'):
        for r in comparisons[method]:gate_table.append(f'|{method}|{r["seed"]}|{percent(r["delta_H_macro"])}|{r["positive_families"]}/{r["defined_families"]}|未通过|')
    ranks=read(root/'A/ranks.json.gz')['rows'];rank_summary=[]
    for method in ('S_LL','G_LL','V_LL','MV_ATTN'):
        for seed in (17,29):
            rs=[r for r in ranks if r['method']==method and r['seed']==seed]
            rank_summary.append(dict(method=method,seed=seed,original_novel_objects=len(rs),defined=sum(r['defined'] for r in rs),
                competition_suppression=sum(r['defined'] and r['top_novel_correct'] and r['full_rank']!=1 for r in rs),
                novel_internal_wrong=sum(r['defined'] and not r['top_novel_correct'] for r in rs),
                top_all_base=sum(r['defined'] and r['top_all_base'] for r in rs),
                novel_only_correct=sum(r['defined'] and r['top_novel_correct'] for r in rs)))
    write(dest/'A/rank_summary.json',dict(status='COMPLETE',rows=rank_summary,causal_mechanism_identified=False))
    rt=['|方法|seed|novel原件|novel内部Top1正确|内部正确但被全词表竞争压制|novel内部错误|全词表Top1为base|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rank_summary:rt.append('|'+ '|'.join(str(r[k]) for k in ('method','seed','original_novel_objects','novel_only_correct','competition_suppression','novel_internal_wrong','top_all_base'))+'|')
    cross=read(root/'A/crossfit.json.gz');ft=['|留出家族|seed17选定、双seed沿用的recipe|原件数|','|---|---|---:|']
    for f in cross['folds']:ft.append(f'|{f["held_family"]}|`{f["recipe"]["candidate_id"]}`|{f["held_originals"]}|')
    repair=read(root/'history/repair_plan.json') if (root/'history/repair_plan.json').exists() else {}
    counter=dict(A_head=0,B_train_validation_head=0,B_engineering_head=0,REG_head=0,C_head=0,cost_head=0,
        new_optimizer_updates=0,replayed_updates=0,dictionary_phi=0,dictionary_Gram=0,solver_iterations=0,
        new_FC_images=0,new_SAM_images=0,new_old_RGB_encodings=0,new_maps=0,new_NQ_encodings=0,new_download_bytes=0,
        new_GPU_workers=0,new_GPU_model_loads=0,**summary['computational_work'])
    counter.update(A_numerical_executions=1+repair.get('prior_A_executions',0),
        total_CPU_candidate_predictions=counter['CPU_candidate_predictions']+repair.get('prior_CPU_candidate_predictions',0),
        total_CPU_CAP_text_matmuls=counter['CPU_CAP_text_matmuls']+repair.get('prior_CPU_CAP_text_matmuls',0),
        total_original_NPZ_imports=counter['original_NPZ_imports']+repair.get('prior_original_NPZ_imports',0))
    write(root/'workload.json',dict(status='COMPLETE',counts=counter,limits=binding['task_specification']['runtime'],
          counter_scope='SCIENTIFIC_PIPELINE; CPU synthetic test work separately in review/test_results.json',
          explanation='A consumes existing arrays; no B/C/REG/cost model worker was activated. CPU calibration is nonzero and separately counted.'))
    shutil.copyfile(root/'workload.json',dest/'workload.json')
    inactive_rows=read(root/'science.json')['phases'];write(dest/'phase_status.json',dict(status='COMPLETE',phases=inactive_rows))
    with (dest/'main_metrics.csv').open('w',newline='') as stream:
        fields=['method','seed','B_macro','N_macro','H_macro','H_micro'];writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow(dict(method=r['method'],seed=r['seed'],**{k:r['metrics'][k] for k in fields[2:]}))
    results='''# Semantic update evidence v1

状态：COMPLETE_A_NO_RESIDUAL_SIGNAL。A真实诊断完成；B/C为NOT_ACTIVATED_BY_A。279原件、215旧base/64旧novel、12家族；270有支持，9个abstention保留且计错。没有缺失原NPZ，不需要重建缓存或补GPU推理。

强简单对照先于复杂方法列出。主表均为clean、8V；S/G/V/MV固定matched1000，FC是同prefix原静态selected列，仅计一次。LOFO每折用另11家族seed17选参，再把同一recipe用于留出家族两个seed，拼回原件后重新按类计算。

'''+ '\n'.join(table)+ '\n\n'+ '\n'.join(gate_table)+'''

B仅在S或G两个seed都ΔH_macro≥1pp、且各≥8个可定义家族正增益时开启。当前均未满足，reference=null。门槛是预定资源政策；不是统计显著性或CVPR接收标准，也不能推断所有S/G差异都已被校准解释。

旧CONF12已曝光，LOFO也仍是探索性诊断。这里没有独立新确认、没有新152/8/40训练，也没有重跑OVI-MAP或全地图AP；不把本表H_macro当原论文AP或系统优越性证明。H_micro由组内普通准确率算；H_macro先按正样本类平均，两者分列。

## Novel错误分解

以下为200类正式竞争和novel-only辅助诊断的差异；novel内部正确但被base等竞争抑制，是描述性计数，不是已识别因果机制。主表有正样本的支持类别为59个旧base和20个旧novel；没有正样本的类别不填0，也不改变200类竞争词表。

'''+ '\n'.join(rt)+ '\n\n## 留一家族选参锁\n\n'+ '\n'.join(ft)+'''

110逻辑候选是55 MIX_GAMMA加55 CAP_GAMMA，不做三重笛卡尔积。gamma使用原余弦单位，MIX不归一化分数。CAP直接读取真实768维单位向量并用原FP32文本重算；未反演embedding。FP64只用于角度几何，裁剪向量回FP32；文本没有再次归一化。原cosine重构最大绝对误差为'''+str(summary['original_score_reconstruction_max_absolute_error'])+'''.

exact_score_alias和exact_prediction_alias分别核对实际数组；相同准确率不构成alias。所有110候选、12折训练候选指标、OOF原件预测、正误路径、全体/base/novel角度、各家族类支持和selected 2/4/8V四条件辅助指标随工件公开。两个seed各自计算，不拼成558个独立对象。

## 条件阶段与资源

|阶段/产物|状态|实测新工作|
|---|---|---:|
|A 导入、排名、110候选、LOFO、决策|COMPLETE|279原NPZ、220组CPU候选预测、10次CPU CAP文本矩阵评分|
|B 角色、数值接入、三臂训练、RAW/POST锁|NOT_ACTIVATED_BY_A|0更新、0head、0字典phi/Gram、0solver迭代|
|B REG26 14槽两融合、成本表|NOT_ACTIVATED_BY_A|0新评分/计时；结果null|
|C 提名、获取、12seed×32mask、REAL/GT预测|NOT_ACTIVATED_BY_A|0下载、0FC/SAM图、0head|
|C 4×1000 bootstrap、最多8张真实案例|NOT_ACTIVATED_BY_A|0抽样集合、0案例|

没有新训练模型或30个checkpoint可交付；未训练模型定义和CPU算子测试不是B真实接入证据。所有B/C公开phase均记录未激活，不能把它们标为运行成功。新地图、N/Q编码、旧RGB编码、GPU模型加载均0；CPU数组操作非零，详见workload.json。实际科学子进程exit=0由外层观察，最终发布后resume另行实测，不预写零工作结论。

## 图表与边界

![主比较](../../../../artifacts/static_ovmap/semantic_update_evidence_v1/figures/A_main.png)

![全110候选](../../../../artifacts/static_ovmap/semantic_update_evidence_v1/figures/A_tradeoffs.png)

![真实向量角度](../../../../artifacts/static_ovmap/semantic_update_evidence_v1/figures/A_angles.png)

![相对FC纠正与误伤](../../../../artifacts/static_ovmap/semantic_update_evidence_v1/figures/A_fix_harm.png)

来源是旧160类监督方案，原novel40已被此前确认观察；新CORE152/PSEUDO8/FINAL40协议本轮没有开启。没有C/REAL/REG新表，也没有系统30FPS、显存或冷启动实测。该结果支持保留简单校准作为强对照，当前没有继续投入E机制新训练的预定残差信号。

'''
    if repair:
        tests=read(root/'review/test_results.json')
        results+=f'\n主审修复公开后备key连接、显式defined屏障，以及cached bind丢失既有消费身份的问题；修复后换producer identity复算。此前候选与OOF数组已保存并逐位核对。{counter["A_numerical_executions"]}次A累计{counter["total_CPU_candidate_predictions"]}组CPU候选预测、{counter["total_CPU_CAP_text_matmuls"]}次CPU CAP文本评分、{counter["total_original_NPZ_imports"]}次原NPZ读取；GPU head始终0。四组最终{tests["final_total"]}项测试通过；初次全组收集遇到控制环境缺timm，T2/T3在既有FC环境运行，T1/T4在控制环境完成。后续仅重跑受影响T1或T4。\n'
    (docs/'RESULTS.md').write_text(results)
    command='python scripts/evaluation/run_semantic_update_evidence_v1.py --parent-root "$STRUCT_ROOT" --output-root "$OUT" --phase all --resume'
    (docs/'HANDOFF.md').write_text('''# Handoff

科学终态：COMPLETE_A_NO_RESIDUAL_SIGNAL。B/C严格未激活；部署N0保持。无需补原缓存：279份原预测NPZ与原文本完整，A读取真实向量而无GPU新推理。

`--parent-root`在本轮唯一表示STRUCT_ROOT；与上轮表示FG_ROOT的参数语义不同。FG/LR/PR/ATTR从STRUCT的已验binding解析，父JSON/缓存只读。已有同任务可恢复，不调用父all，不扫描全历史。

本地继续：
```bash
'''+command+'''
```

控制环境继承父ovimap-map Python；BLAS1。A不加载FC/SAM；GPU若指定必须等于继承GPU0，不能擅换卡。公开主数据：A/source_scores.npz为完整原FP32分数、defined、valid_ids；A/source_vectors.npz为9个主列clean8V的真实768向量和原文本，original_column_indices按全局columns连接。A/import.json含全部key/类/列、源SHA与类别分割；A/predictions.npz含110候选与OOF预测。可由replay.py重建A输入并复算；不存在从200分数伪逆恢复向量。

B/C阶段入口在当前闭合gate下统一返回NOT_ACTIVATED_BY_A；B模型定义/算子为未训练代码，不声明已验证B/C服务器接入。若未来科学合同要改变门槛、输入或数学，必须新identity并重新审计，不能沿用本轮负结果收据宣称完成新实验。

没有外部全方法复现、主分支合并或部署变更。C1固定成果，C2仅C1收据与永久链接。C2 SHA及真实零工作resume结果写OUT/publication，不产生第三次自引用提交。
''')
    (docs/'DATA_SCOPE.md').write_text('''# Data scope

A：旧CONF12=已曝光探索；279个原件，215旧base、64旧novel，9未知仍计错，270支持子集另列。每方法同一原件分母；每类仅有正样本才进入macro。每家族支持类集合不同，跨家族H不能简单平均替代池化原件H。

标签用于A排名/计分及预定资源选择，不宣称独立确认。旧TRAIN/DEV/REG没有新监督或新评分。两seed各自评估，同场景、四条件均相关，不当作新独立样本。静态FC一次，不按两个seed虚增权重。

B/C完全未激活，CORE152/PSEUDO8角色、12个新家族、REAL提案与新GT诊断分母均未生成。无新152类CE或新数据获取。公开仅派生分数、向量/文本、诊断指标与原件key/类角色；不上传raw RGB、mesh、逐顶点GT、优化器、基础权重或凭证。
''')
    (docs/'EXTERNAL_CONTROL_SCOPE.md').write_text('''# External controls

EXTERNAL_FULL_METHOD_REPRODUCTION=NOT_RUN。110候选是本项目固定分数混合/偏置/角度对照，不是WiSE-FT、CAMFusion或Mask-Adapter复现。

[WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)采用模型权重组合；本轮MIX组合原冻结FC和MV输出分数，未运行其完整训练/评估。

[GZSL校准](https://arxiv.org/abs/1605.04253)研究已见/未见类竞争校准；gamma类偏置是强简单控制，不能作为广泛新颖性主张。

[Mask-Adapter官方仓库](https://github.com/hustvl/MaskAdapter)是mask特征适配相关近邻；本轮没有执行其外部协议。有限FISTA/convex evidence定义尚未激活真实B实验；归一化后的向量一般不在原欧氏凸包，不声称精确交集投影。公开来源包括CVF检索记录、[WiSE-FT原作者预印本](https://arxiv.org/abs/2109.01903)、GZSL原论文和Mask-Adapter官方仓库；CVF直接抓取出现超时/403，没有把这些失败写成读过PDF全文。不是穷尽最新文献查新或CVPR录用判断。
''')
    (docs/'CLAIMS_LEDGER.md').write_text('''# Claims ledger

|主张|状态|直接证据/边界|
|---|---|---|
|本轮A所需缓存不存在|不成立|A/import.json：279原NPZ、原text均已验；0补head|
|S/G超过固定MV|成立于曝光GT探索|主表matched1000；不等于超过强简单对照|
|S/G保有达到预定门槛的残差优势|未支持|A/decision.json：两个seed及8-family规则未过|
|MV错误全由base偏置造成|未证明|ranks逐原件分解；novel内部仍有错误|
|简单校准已解释所有机制差异|未证明|LOFO仅探索敏感性，不是因果识别|
|E有限算子优于RADIUS或SIMPLE_STAR|未测试|B未激活；CPU合成测试不填科研表|
|已完成新场景独立泛化确认|未测试|C未激活；原40类已历史曝光|
|已整体优于OVI-MAP或达到30FPS|未支持|无本轮全方法复现/系统时间与VRAM测试|
|这是CVPR充分证据/保证录用|不作主张|A负资源门槛结论，非投稿科学认证|
''')
    files=[p for p in dest.rglob('*') if p.is_file() and p.name!='artifact_index.json']
    manifest=write(dest/'artifact_index.json',dict(status='COMPLETE',files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=digest_file(p)) for p in sorted(files)],
          total_bytes=sum(p.stat().st_size for p in files),maximum_blob_bytes=max(p.stat().st_size for p in files),
          raw_images_mesh_GT_optimizer_baseweights_published=False))
    return write(root/'report.json',dict(status=science['status'],summary_identity=summary['identity'],
          scientific_actual_exit_code=observed['actual_exit_code'],artifact_index=manifest['identity'],
          docs=[str(DOCS/n) for n in ('RESULTS.md','HANDOFF.md','REVIEW.md','CLAIMS_LEDGER.md','DATA_SCOPE.md','EXTERNAL_CONTROL_SCOPE.md')]))
