"""Reports read measured receipts only; missing required stages remain explicit."""
from pathlib import Path
import json
from .common import REPO,verified,write,ConsumptionIndex,canonical_digest,read_document,METHODS,STATICS

DOCS=Path('docs/paper/static_ovmap/structure_confirmation_v1')
ARTIFACTS=Path('artifacts/static_ovmap/structure_confirmation_v1')
REQUIRED={'preparation':'prepare_cache.json','training':'training/receipt.json','selection':'model_selection_lock.json','predictors':'predictor_lock.json','raw_confirmation':'confirmation/acquisition.json','confirmation_geometry':'confirmation/geometry_lock.json','confirmation_inputs':'confirmation/real_input_lock.json','confirmation_features':'confirmation/features/manifest.json','real_predictions':'confirmation/real_prediction_lock.json','GT_inputs':'confirmation/GT/input_lock.json','GT_predictions':'confirmation/GT/prediction_lock.json','confirmation_score':'confirmation/score.json','map_predictions':'map/prediction_lock.json','map_score':'map/score.json','statistics':'statistics/receipt.json','cost':'cost/receipt.json','qualitative':'figures/receipt.json'}

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)

def number(value,percent=True,digits=2):return 'null' if value is None else f'{value*(100 if percent else 1):.{digits}f}'

def run(binding,*,phase='report'):
    root=Path(binding['output_root']);docs=REPO/DOCS;docs.mkdir(parents=True,exist_ok=True);records={};missing=[]
    if (root/'training/receipt.json').exists():
        from .accounting import summarize
        detail=summarize(binding)
    else:detail=None
    for name,path in REQUIRED.items():
        if not(root/path).exists():missing.append(dict(stage=name,path=path,status='NOT_COMPLETED'));continue
        value=verified(root/path);records[name]=value
        if value['status']!='COMPLETE':missing.append(dict(stage=name,path=path,status=value['status']))
    observed=verified(root/'execution_observed.json') if(root/'execution_observed.json').exists() else None
    if observed is None or observed['actual_exit_code']!=0:missing.append(dict(stage='observed_scientific_exit',value=observed))
    primary=verified(root/'review/primary_review.json') if(root/'review/primary_review.json').exists() else None
    if primary is None or primary['status']!='COMPLETE':missing.append(dict(stage='primary_review',status='PENDING'))
    status='COMPLETE' if not missing else 'PARTIAL';selection=records.get('selection');rows=[]
    if selection:
        for key,arm in selection['arms'].items():
            for profile in ('matched1000','selected','terminal2000'):
                r=arm[profile];rows.append([arm['method'],arm['seed'],profile,r['step'],number(r['A']),number(r['M']),number(r['C'])])
    result='# FG2结构验证与确认\n\n状态：'+status+'。指标以百分比呈现，差值以百分点呈现；缺失结果为null。\n\n'
    if primary and primary.get('scientific_interpretation'):
        result+='\n\n'.join(primary['scientific_interpretation'])+'\n\n'
    if (root/'confirmation/GT/score.json').exists() and (root/'confirmation/real_score.json').exists():
        gt_main=verified(root/'confirmation/GT/score.json');real_main=verified(root/'confirmation/real_score.json');main_rows=[]
        def mean_defined(values):return sum(values)/len(values) if values and all(v is not None for v in values) else None
        for method in (*METHODS,*STATICS):
            static=method in STATICS
            gm=[gt_main['metrics'][str(i)]['full_originals'] for i,c in enumerate(gt_main['columns']) if c['method']==method and c['prefix']==8 and (static or c['profile']=='matched1000')]
            rm=[r['metrics'] for r in real_main['pools'] if r['column_spec']['method']==method and (static or r['column_spec']['profile']=='selected')]
            values=[mean_defined([m[part][metric] for m in gm]) for part,metric in [('all','ordinary_top1'),('all','balanced_top1'),('base','ordinary_top1'),('novel','ordinary_top1')]]
            values.extend(mean_defined([m[metric] for m in rm]) for metric in ('apall','ap50','ap25'))
            step='静态' if static else selection['arms'][method+':seed17']['selected']['step']
            main_rows.append([method,step,*[number(v) for v in values]])
        result+='## 核心对照\n\nGT固定1000步，真实提案使用各方法预锁定DEV选择步；均为8视图上限。学习方法先分别评分两seed再平均，静态方法只测一次；不拼接seed，也不平均scene AP。\n\n'+table(['方法','REAL选步','GT Top1 (%)','GT balanced (%)','GT base (%)','GT novel (%)','REAL APall (%)','REAL AP50 (%)','REAL AP25 (%)'],main_rows)+'\n'
    result+='## DEV：匹配步、选择步与终点\n\n'+table(['方法','seed','口径','step','A (%)','M (%)','C (%)'],rows)+'\n'
    result+='DEV包含69个base原件、4个家族、四条件，竞争词表为200类。仅seed17按A/M/C及最早步选择；seed29沿用同方法seed17步数。\n\n'
    if (root/'review/dev_key_repair.json').exists():
        result+='开发期间在G首个DEV前修正了base过滤后的排列键索引。S/V的本任务既有权重与DEV分数未改，模型/优化器/RNG/采样内容等价证明及原始记录保留；G从安全断点重放，实际重放另计。详见review/dev_key_repair_contract.json与workload记录。\n\n'
    if (root/'review/sam_linker_isolation.json').exists():
        result+='确认阶段修复了控制环境抢先加载CUDA库导致的SAM导入失败，以及地图payload的禁用元数据字段。两次SAM失败均在import torch、进入main前终止，新增编码为0；失败日志与身份保留。修复仅隔离SAM子进程链接库并移除payload中的GT_used声明，未改变SAM/FC数值、已锁模型或地图标签计算。训练续跑核验原16臂与80检查点后直接复用原汇总，不覆盖实际训练消耗。\n\n'
    if (root/'engineering/real_G_layout.json').exists():
        layout=verified(root/'engineering/real_G_layout.json')
        result+='真实DEV上S/G的'+str(layout['actual_DEV_condition_pairs'])+'组条件记录具有相同对象键、视图、group数和有效观测数量。G的实际同槽保留率如下，未强制排列必须改变每个token：\n\n'+table(['seed','有效token','仍在原槽token','同槽保留率 (%)'],[[r['seed'],r['valid_tokens'],r['retained_same_slot_tokens'],number(r['retained_same_slot_fraction'])] for r in layout['training_permutation_audits']])+'\n'
    if detail:
        result+='## 训练与投影消耗\n\n'+table(['方法','seed','updates','名义head forward','phi forward','phi向量','完成臂秒(含DEV)','参数','可训练参数'],[[r['method'],r['seed'],r['completed_updates'],r['nominal_head_forwards'],r['nominal_phi_forward_calls'],r['nominal_phi_projected_vectors'],number(r['elapsed_seconds'],False),r['parameters'],r['trainable_parameters']] for r in detail['rows']])+'\nphi次数由实际draw与已验缓存valid布局精确推导，单列为源码推导证据；未测得逐kernel训练耗时，填null。每臂用时含训练、DEV、缓存读取和断点写入；中断回退前的额外耗时未单独计时，不能把各臂用时之和当作全部尝试的总训练时间。物理head和重放以总计数器为准。\n\n'
    if (root/'engineering/G_correspondence.json').exists():
        correspondence=verified(root/'engineering/G_correspondence.json')
        result+='G分组中同一规范物理位置的跨视图配对残留，直接从已保存排列计算；不含模型seed/step重复，不新增head：\n\n'+table(['输入域','条件序号','views上限','跨视图对','保留物理对','残留率 (%)','singleton组'],[[r['domain'],r['condition'],r['prefix'],r['cross_view_pairs'],r['same_physical_site_pairs'],number(r['retained_physical_pair_fraction']),r['singleton_groups']] for r in correspondence['rows']])+'\nsingleton不进入配对分母；同槽token保留率与跨视图物理配对残留率是不同统计量。\n\n'
    CONF=[];coverage=None
    if (root/'confirmation/real_score.json').exists():
        real=verified(root/'confirmation/real_score.json')
        for r in real['pools']:
            c=r['column_spec'];CONF.append([c['method'],c['seed'],c['profile'],c['step'],number(r['metrics']['apall']),number(r['metrics']['ap50']),number(r['metrics']['ap25'])])
        coverage=real['recognition_coverage'];result+='## 新家族真实提案\n\n'+table(['方法','seed','口径','step','APall (%)','AP50 (%)','AP25 (%)'],CONF)+'\n'
        result+='固定重建表面上的稀疏SAM提案，完整GT分母、原官方matcher与池化；所有方法共享支持及constant1排序。该表不代表新场景在线建图。\n\n'
        if coverage:
            counts=coverage[0]['rows'];result+='共享几何支持：\n\n'+table(['IoU >','匹配提案','唯一GT','family','heldout唯一GT','heldout family'],[[r['IoU_strictly_greater'],r['matched_proposals'],r['unique_GT'],r['families'],r['heldout_unique_GT'],r['heldout_families']] for r in counts])+'\n'
            support=next(r for r in counts if r['IoU_strictly_greater']==.5)
            limited=support['unique_GT']<40 or support['families']<8 or support['heldout_unique_GT']<20 or support['heldout_families']<4
            result+='预登记覆盖参考：'+('LIMITED_REAL_OR_NOVEL_SUPPORT' if limited else 'MEETS_PREREGISTERED_RESOURCE_COVERAGE_REFERENCE')+'。40个唯一GT/8家族、20个heldout唯一GT/4家族只是资源覆盖参考，不是显著性或成功门槛。\n\n'
    else:result+='## 新家族真实提案\n\nAPall/AP50/AP25：null；尚无完整确认评分。\n\n'
    if (root/'confirmation/GT/score.json').exists():
        gt=verified(root/'confirmation/GT/score.json');gr=[]
        for ci,c in enumerate(gt['columns']):
            m=gt['metrics'][str(ci)]['full_originals'];gr.append([c['method'],c['seed'],c['profile'],c['step'],c['prefix'],number(m['all']['ordinary_top1']),number(m['all']['balanced_top1']),number(m['base']['ordinary_top1']),number(m['novel']['ordinary_top1']),number(m['H_base_novel']),m['all']['original_objects'],m['all']['abstention_n']])
        result+='## 独立GT支持诊断\n\n'+table(['方法','seed','口径','step','views','Top1','balanced','base','novel','H','原件','abstention条件'],gr)+'\n未观察原件保留主分母；heldout CE160为null。GT派生支持与真实SAM支持分别评价。\n\n'
        first=gt['metrics']['0']['full_originals'];result+='固定原件覆盖：\n\n'+table(['分组','原件','family','正样本类别','四条件记录'],[[part,first[part]['original_objects'],first[part]['families'],first[part]['classes'],first[part]['variant_records']] for part in ('base','novel','all')])+'\n'
        loss_rows=[]
        for ci,c in enumerate(gt['columns']):
            if c['prefix']!=8:continue
            for part in ('base','novel','all'):
                m=gt['metrics'][str(ci)]['full_originals'][part];loss_rows.append([c['method'],c['seed'],c['profile'],part,number(m['NLL200'],False),m['NLL200_defined_n'],number(m['CE160'],False),m['CE160_defined_n'],m['abstention_n']])
        result+='全200竞争的损失及定义分母：\n\n'+table(['方法','seed','口径','分组','NLL200','defined_n','CE160','defined_n','abstention条件'],loss_rows)+'\n'
        family_rows=[];heldout=set(verified(root/'class_split.json')['heldout_ids']);real_matches=verified(root/'confirmation/real_score.json')['proposal_matches']
        for scene in verified(root/'role_lock.json')['roles']['CONF']:
            family=scene.split('_')[0];originals=[r for r in gt['records']['0'] if r['family']==family];values=[]
            for method in ('S_LL','G_LL','V_LL','MV_ATTN'):
                points=[gt['metrics'][str(i)]['per_family'][family]['all']['ordinary_top1'] for i,c in enumerate(gt['columns']) if c['method']==method and c['profile']=='matched1000' and c['prefix']==8]
                values.append(number(sum(points)/len(points)) if points and all(p is not None for p in points) else 'null')
            matched=[r for r in real_matches if r['family']==family and r['class_id'] is not None and r['IoU']>.5]
            family_rows.append([family,len(originals),sum(not r['base'] for r in originals),*values,len({(r['scene'],r['GT_owner']) for r in matched}),len({(r['scene'],r['GT_owner']) for r in matched if r['class_id'] in heldout})])
        result+='每家族GT诊断clean Top1（matched1000、两seed点估计均值，原件主分母）及真实提案覆盖：\n\n'+table(['family','GT原件','novel原件','S_LL (%)','G_LL (%)','V_LL (%)','MV_ATTN (%)','真实IoU>.5唯一GT','其中novel'],family_rows)+'\n每seed及各family的完整分类细项保留于[GT评分记录](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/confirmation/GT/score.json.gz)。家族数是统计聚类数，不将两seed作为双倍样本。\n\n'
    if 'map_score' in records:
        maps=records['map_score'];mr=[]
        for cohort,values in maps['pools'].items():
            for name,r in values.items():mr.append([cohort,name,*[number(r['metrics'][m]) for m in ('apall','ap50','ap25','miou','macc')]])
        result+='## 已曝光26场景完整地图回归\n\n'+table(['cohort','方法与读出','APall','AP50','AP25','mIoU','mAcc'],mr)+'\n固定G1分区、旧eligible/NQ/温度及scene-specific词表。该回归不增加独立确认家族。\n\n'
    if 'statistics' in records:
        sr=[]
        for pair in records['statistics']['pairs']:
            for metric,value in pair['intervals'].items():
                includes_zero='null' if value['lower_025'] is None or value['upper_975'] is None else '是' if value['lower_025']<=0<=value['upper_975'] else '否'
                sr.append(['/'.join(pair['pair']),metric,number(pair['full_delta'][metric],digits=4),number(value['lower_025'],digits=4),number(value['upper_975'],digits=4),includes_zero,value['defined_draws'],value['undefined_draws']])
        result+='## 固定家族配对统计\n\n'+table(['S减对照','metric','均值差(pp)','2.5%','97.5%','区间含0','defined','undefined'],sr)+'\n四对共4000个预定抽样集合。先分别测两seed再平均差；GT主口径为1000步，真实AP主口径为DEV选择步。各指标报告未作多重比较调整的percentile家族bootstrap区间；边界保留4位小数，避免微小负值被显示为0.00。\n\n'
        factors=verified(root/'statistics/factorial_CONF.json');fr=[]
        for r in factors['rows']:fr.append([r['scope'],r['seed'],r['profile'],r['metric'],number(r['inner_relative_mean_effect']),number(r['outer_relative_mean_effect']),number(r['interaction']),number(r['LL_minus_UU'])])
        result+='## 确认集两级相对权重效应\n\n'+table(['任务','seed','口径','metric','inner均值效应(pp)','outer均值效应(pp)','interaction(pp)','LL−UU(pp)'],fr)+'\nUU仍学习自己的总路由质量；interaction单列，不与两个主效应重复相加。\n\n'
    if 'cost' in records:
        cr=[]
        for r in records['cost']['summaries']:cr.append([r['slot'],r['prefix'],number(r['total_seconds_median']*1000,False),number(r['total_seconds_p90']*1000,False),number(r['head_seconds_median']*1000,False),number(r['peak_allocated_bytes_p90']/2**30,False),r['parameters']])
        result+='## 实测成本\n\n'+table(['模型','views','含pool/phi/transfer median ms','P90 ms','head median ms','allocated P90 GiB','参数'],cr)+'\n模型已驻留、读取已缓存FC grid；head计时包含在总计时中。该表不是冷启动或系统30FPS证明。\n\n'
        shared=[]
        if (root/'confirmation/SAM/generator_lock.json').exists():
            sam=verified(root/'confirmation/SAM/generator_lock.json');shared.append(['SAM2 AMG',sam['counts']['image_encoder_calls'],sam['counts']['mask_decoder_batches'],number(sam['elapsed_seconds'],False),number(sam['peak_allocated_bytes']/2**30,False),'含模型构造/加载及固定seed图提案，不含环境启动；全部方法共享一次'])
        if 'confirmation_features' in records:
            fc=records['confirmation_features'];shared.append(['FC全图捕获',fc['physical_encodings'],'不适用',number(fc['elapsed_seconds'],False),number(fc['peak_allocated_bytes']/2**30,False),'完整固定bank捕获，含逐帧预处理、提取、投影与缓存写入'])
        result+='新确认输入的共享计算只发生一次，不逐方法重复相加：\n\n'+table(['共享模块','图像编码','decoder batch','阶段秒','allocated峰值 GiB','边界'],shared)+'\n'
    if 'qualitative' in records:
        figures=records['qualitative'];figure_rows=[]
        captions={'grouping':'CONF12真实稀疏提案池化APall (%)；DEV选择步、8V；两seed分别显示，静态只计一次。','factorial':'已曝光DEV A (%)；固定1000步，左右为seed17/29；两个面板共享颜色范围，格内为实测值。','budget':'左：全部8方法的DEV A (%)；右：CONF固定GT原件的clean Top1 (%)、选择步与2/4/8视图前缀；实线seed17、虚线seed29。','coverage':'左：真实AMG/提案/唯一GT的不同计数单位；右：固定base/novel GT原件的支持与abstention，不混用两个任务分母。'}
        for name in figures['statistical_figures']:
            figure_rows.append([name,'[PNG](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/'+name+'.png) · [PDF](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/'+name+'.pdf)',captions[name]])
        result+='## 图与实际案例\n\n'+table(['图','文件','内容与口径'],figure_rows)+'\n'+table(['对照','分层','实际案例','缺少位置'],[['/'.join(row['pair']),row['stratum'],len(row['cases']),row['unavailable_slots']] for row in figures['cases']])+'\n案例来源和确定性选取记录见[来源JSON](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/figures/source_data.json.gz)。案例使用实测深度/mesh投影，三列共享相机、尺度和颜色；原始RGB不上传。\n\n'
    workload=verified(root/'workload.json') if(root/'workload.json').exists() else None
    result+='## 覆盖、消耗与缺项\n\n'+table(['阶段','状态'],[[k,records[k]['status'] if k in records else 'NOT_COMPLETED'] for k in REQUIRED])+'\n'
    result+='```json\n'+json.dumps(dict(workload=workload,missing=missing),ensure_ascii=False,indent=2)+'\n```\n\n部署维持N0。所有控制、负结果和undefined单元均保留；本报告不据单项上涨声称整体优于OVI-MAP或达到CVPR录用标准。\n'
    (docs/'RESULTS.md').write_text(result)
    roles=verified(root/'role_lock.json')
    from .common import family
    (docs/'DATA_SCOPE.md').write_text('# 数据范围\n\n'+table(['角色','精确场景','物理家族数','家族'],[[k,', '.join(v),len({family(s) for s in v}),', '.join(sorted({family(s) for s in v}))] for k,v in roles['roles'].items()])+'\nTRAIN/DEV/旧地图均已曝光。CONF名单来自固定Git blob `'+roles['confirmation_plan_blob']+'`，角色来源与原提名排除依据见[家族派生核对](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/review/role_family_derivation.json.gz)。该核对只补充既有锁的可读元数据，未修改或重选角色。CONF仅证明未参与当前开发；完整历史使用和预训练独立性未证明。真实提案来自提供的无语义重建；GT派生支持只作独立诊断。\n')
    (docs/'EXTERNAL_CONTROL_SCOPE.md').write_text('# 外部对照范围\n\nEXTERNAL_FULL_REPRODUCTION=NOT_RUN。\n\nMV_ATTN是在固定region输入上训练的256维、4层受控多视图attention模型，使用本轮父CE目标。[CAMFusion v1](https://arxiv.org/html/2604.12551v1)采用不同的自注意力/跨视图记忆交互、1024维8层模型与额外多视图对比目标。本轮仅检验受控学习式融合，不能把结果当作原版CAMFusion复现。\n\nB_PFE_ADAPTED仅在当前对象采样位置做可见性区域特征聚合。[Open3DIS v3 §3.3](https://arxiv.org/html/2312.10671v3)还包含跨提案的点特征累积、原多尺度裁剪与提案流程；[作者仓库](https://github.com/VinAIResearch/Open3DIS)的完整方法未在本轮运行。不能据此表述超越两篇原论文的完整系统。原始设置复现需要单独协议与预算。\n')
    claims=[['同输入真实对应是否有效','以新S/G matched1000与selected确认及区间为准','不以旧推理shuffle替代从头训练G'],['两级学习是否必要','LL/UL/LU/UU与普通表面均值分别报告','UU保留可学习mass，不是无训练均值'],['新家族泛化','只限封存CONF12与实际base/novel/real支持','覆盖不足保留，不换场景'],['实例与语义','新提案共享几何；旧26报告完整图指标','不声称几何关联/在线建图提升'],['效率','驻留模型下pool/phi/transfer/head实测','不从缓存回放推出系统30FPS'],['投稿结论','状态与证据范围分开','无录用概率或整体SOTA承诺']]
    (docs/'CLAIMS_LEDGER.md').write_text('# 主张与证据边界\n\n'+table(['问题','可用证据','结论限制'],claims))
    if primary and primary.get('scientific_interpretation'):
        interpretation='\n## 根据实测证据的判断\n\n'+'\n\n'.join(primary['scientific_interpretation'])+'\n'
        for name in ('CLAIMS_LEDGER.md',):
            p=docs/name;p.write_text(p.read_text()+interpretation)
    review='# 最终合同核对\n\n'+table(['合同项','当前证据'],[[k,REQUIRED[k]+'：'+records[k]['status'] if k in records else REQUIRED[k]+'：缺失'] for k in REQUIRED])+'\n主代理审核收据：'+('review/primary_review.json' if primary else 'PENDING')+'。实际科学进程退出：'+str(observed['actual_exit_code'] if observed else None)+'。\n\n'+json.dumps(missing,ensure_ascii=False,indent=2)+'\n'
    if primary:
        review+='\n[主代理完整核对记录](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/review/primary_review.json.gz)\n\n'+table(['提示词条款','核对状态','具体证据'],[[r['clause'],r['status'],r['evidence']] for r in primary.get('contract_checks',[])])
    (docs/'REVIEW.md').write_text(review)
    handoff='# FG2结构验证与确认交接\n\n当前状态：'+status+'。\n\n```bash\ncd "'+str(REPO)+'"\n/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_fg2_structure_confirmation_v1.py --output-root "'+str(root)+'" --phase all --resume\n```\n\n--parent-root仅代表FG_ROOT；继承parent_root与lr_source_root语义未改。实际FG_ROOT=`'+binding['fg_parent_root']+'`，LR_ROOT=`'+binding['lr_parent_root']+'`。原数据与训练缓存只读；新数据和断点保存在独立OUT。小头/无损cosines索引位于artifacts/static_ovmap/structure_confirmation_v1，architecture.json与weights/manifest.json提供结构、参数与加载接口。\n\nSAM子进程通过本任务sam_runtime.py隔离nvJitLink，不修改原环境。完整终态续跑验证数据/源码与远端身份后不启动GPU worker、不新增head或提交。发布收据生成后，本文件追加C1永久链接；C2精确SHA保存在外部OUT/publication/final.json，避免自引用提交。\n'
    handoff+='\n加载公开model-only小头并验证原权重字节（在仓库根目录执行，不启动GPU）：\n\n```bash\nPYTHONPATH=src:. /home/ww/miniconda3/envs/ovimap-map/bin/python - <<\'PY\'\nfrom pathlib import Path\nfrom static_ovmap.structure_confirmation_v1.common import verified\nfrom static_ovmap.structure_confirmation_v1.export import load_export\nfolder = Path("artifacts/static_ovmap/structure_confirmation_v1")\nmanifest = verified(folder / "weights/manifest.json")\nkey = manifest["profiles"]["S_LL:seed17:selected"]\nhead, state = load_export(folder, key, device="cpu")\nprint(state["method"], state["seed"], state["step"], key)\nPY\n```\n\n该加载接口恢复小头；实际前向仍需锁定FC的phi及精确七张量输入。基础FC权重和原始ScanNet数据未上传。\n'
    if (root/'publication/C1.json').exists():handoff+='\nC1已验证收据：publication_receipt.json。\n'
    (docs/'HANDOFF.md').write_text(handoff)
    from .export import export
    artifacts=export(binding,REPO/ARTIFACTS) if selection else write(REPO/ARTIFACTS/'artifact_index.json',dict(status='PARTIAL',reason='No completed selection; no trained weights fabricated',files=[]))
    if artifacts['status']!='COMPLETE':
        status='PARTIAL';missing.append(dict(stage='artifact_export',status=artifacts['status']))
        for name in ('RESULTS.md','HANDOFF.md'):
            p=docs/name;p.write_text(p.read_text().replace('状态：COMPLETE','状态：PARTIAL'))
    return write(root/'report.json',dict(status=status,required={k:v['identity'] for k,v in records.items()},missing=missing,artifact_index=artifacts['identity'],workload=workload,coverage=coverage,scientific_exit=observed,primary_review=primary['identity'] if primary else None,publication_status='NOT_ATTEMPTED'))
