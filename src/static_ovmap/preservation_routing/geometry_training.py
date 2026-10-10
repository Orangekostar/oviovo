"""Five paired controls on separately qualified frozen R bases."""
from pathlib import Path
from .common import method,objects,verified,write
from .training import train_arm,draw_schedule,model_digest
from .recognition import evaluate,load_head
from .teachers import load
from .selection import choose

def train_G(binding,fc,loader,seed):
    root=Path(binding['output_root'])
    nominations={s:verified(root/f'R_nomination_seed{s}.json') for s in (17,29)}
    if any(n['Rstar'] is None for n in nominations.values()):raise ValueError('Both R seeds must qualify before any G update')
    if nominations[17]['Rstar']!=nominations[29]['Rstar']:raise ValueError('Repeat R architecture switched')
    name=nominations[seed]['Rstar'];source=nominations[seed]['checkpoints'][name]
    base,state=load_head(source['path'],fc.device)
    directory=root/'training'/f'seed{seed}';ref_path=root/f'recognition/dev_Rstar_seed{seed}.json'
    reference=verified(ref_path) if ref_path.exists() else evaluate(base,loader,objects(binding,'dev'),fc,save_path=ref_path)
    teachers=verified(root/'teachers/manifest.json');teachers=dict(teachers,cache=load(teachers,fc.device))
    train=objects(binding,'train');dev=objects(binding,'dev');schedule=draw_schedule(train,seed)
    base_spec=dict(config=state['config'],checkpoint=source)
    rows=[train_arm(binding,fc,loader,teachers,train,dev,schedule,seed,cfg,reference,base=base_spec)
          for cfg in binding['specification']['G_methods']]
    if len({r['initial_model_digest'] for r in rows})!=1:raise ValueError('Matched G initial states differ')
    candidates=[dict(r['selected_metrics'],branch=r['branch'],order=i,checkpoint=r['selected_checkpoint']) for i,r in enumerate(rows) if r['foundation_pass']]
    winner=choose(candidates)
    nominee=(winner['branch'] if winner else name) if seed==17 else verified(root/'G_nomination_seed17.json')['architecture']
    return write(root/f'G_nomination_seed{seed}.json',dict(status='NOMINATED',seed=seed,architecture=nominee,
              Rstar=name,Rstar_checkpoint=source,checkpoints={r['branch']:r['selected_checkpoint'] for r in rows},
              arms={r['branch']:r['selected_metrics'] for r in rows},scientific_updates=10000,
              selected_before_transfers=True,repeat_all_five=seed==29,repeat_architecture_switch=False))
