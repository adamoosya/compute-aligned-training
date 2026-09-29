"""Explicit SFT stages for the token-margin and SFT-to-RL appendix experiments."""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import argparse,json,math,os,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from cat.data import CausalCollator,encode_math_selection,assert_disjoint,load_math_hf
from cat.data.math import load_math_jsonl
from cat.experiments.sft_config import SFTRunConfig,SFTTrainConfig,MathDataConfig
from cat.models.hf import HFModelConfig,LoRASettings,load_lora_model,save_lora_adapter,require_revision,_adapter_metadata,environment_versions
from cat.paper.reference import read_json
from cat.training import SFTObjective,train_sft_epoch
from cat.training.margin import train_margin_epoch

@dataclass(frozen=True)
class AppendixSFTConfig:
    name:str
    base:SFTRunConfig
    initialization:str
    objective:SFTObjective
    margin_weight:float
    rivals:int
    export_stage:str
    source_note:str
    schema_version:int=1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1:raise ValueError('Invalid schema')
        if not isinstance(self.name,str) or not self.name:raise ValueError('Name required')
        if self.initialization not in {'base','adapter'}:raise ValueError('Invalid initialization')
        if self.export_stage not in {'warmup','branch'}:raise ValueError('Invalid export stage')
        if type(self.rivals) is not int or self.rivals<1:raise ValueError('Invalid rival count')
        if isinstance(self.margin_weight,bool) or not isinstance(self.margin_weight,(int,float)) or not 0<=self.margin_weight<=1:raise ValueError('Invalid margin weight')
        if self.margin_weight and (self.objective.strategy!='ce' or self.objective.reduction!='token_mean'):
            raise ValueError('Pairwise hybrid uses token-mean CE')
        if not isinstance(self.source_note,str) or not self.source_note:raise ValueError('Source note required')


def load_appendix_sft(path):
    obj=read_json(path)
    if set(obj)!=set(AppendixSFTConfig.__dataclass_fields__):raise ValueError('Missing or unknown appendix fields')
    b=obj['base'].copy()
    try:
        for k,cls in [('model',HFModelConfig),('lora',LoRASettings),('data',MathDataConfig),('objective',SFTObjective),('training',SFTTrainConfig)]:b[k]=cls(**b[k])
        obj['base']=SFTRunConfig(**b);obj['objective']=SFTObjective(**obj['objective'])
        return AppendixSFTConfig(**obj)
    except (TypeError,KeyError) as e:raise ValueError(f'Invalid appendix config: {e}') from e


def run_appendix(config,output,*,initial_adapter=None,allow_download=False,local_data=None,cache_dir=None):
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Single-process only')
    root=Path(output).expanduser()
    if root.exists() or root.is_symlink():raise FileExistsError('Use a new output directory')
    b=config.base
    b.model.preflight()
    if (config.initialization=='adapter') != (initial_adapter is not None):raise ValueError('Initial adapter does not match requested stage')
    prior=_adapter_metadata(initial_adapter,b.model,b.lora) if initial_adapter else None
    d=b.data
    kwargs=dict(split=d.split,levels=d.levels,max_examples=d.max_examples,seed=d.seed)
    if local_data:
        selection=load_math_jsonl(local_data,**kwargs)
    else:
        require_revision(d.revision,'data.revision')
        import datasets
        previous=datasets.config.HF_DATASETS_OFFLINE
        try:
            if not allow_download:datasets.config.HF_DATASETS_OFFLINE=True
            selection=load_math_hf(dataset_name=d.dataset_name,revision=d.revision,cache_dir=cache_dir,**kwargs)
        finally:datasets.config.HF_DATASETS_OFFLINE=previous
    assert_disjoint(selection)
    manifest=selection.manifest()
    if prior is not None:
        origin=prior.get('provenance',{})
        if origin.get('stage')!='warmup' or origin.get('selection_sha256')!=manifest['selection_sha256']:
            raise ValueError('Token-margin branches need the matching shared warmup and selection')
        if origin.get('target')!=d.target:raise ValueError('Warmup target mismatch')
    random.seed(b.training.seed);torch.manual_seed(b.training.seed)
    model,tokenizer=load_lora_model(b.model,b.lora,initial_adapter=initial_adapter,allow_download=allow_download,cache_dir=cache_dir)
    encoded=encode_math_selection(selection,tokenizer,target=d.target,max_length=d.max_length,overflow=d.overflow,prompt_template=d.prompt_template)
    root.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    write('config.json',asdict(config));write('selection.json',manifest);write('encoding.json',encoded.manifest())
    write('environment.json',environment_versions())
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=b.training.learning_rate,weight_decay=b.training.weight_decay)
    total_steps=b.training.epochs*math.ceil(math.ceil(len(encoded)/b.training.microbatch_size)/b.training.accumulation_steps)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:max(0.,1-step/total_steps))
    scaler=torch.amp.GradScaler('cuda') if b.model.precision=='fp16' else None
    generator=torch.Generator().manual_seed(b.training.seed)
    metrics=[]
    for epoch in range(b.training.epochs):
        loader=DataLoader(encoded,batch_size=b.training.microbatch_size,shuffle=True,generator=generator,num_workers=0,collate_fn=CausalCollator(tokenizer.pad_token_id))
        shared=dict(accumulation_steps=b.training.accumulation_steps,max_grad_norm=b.training.max_grad_norm,
                    device=b.model.device,precision=b.model.precision,scaler=scaler,scheduler=scheduler)
        if config.margin_weight:
            stat=train_margin_epoch(model,loader,opt,margin_weight=config.margin_weight,rivals=config.rivals,**shared)
        else:stat=train_sft_epoch(model,loader,opt,config.objective,**shared)
        metrics.append(asdict(stat));write(f'metrics-{epoch+1:03d}.json',metrics[-1])
    provenance={'stage':config.export_stage,'run_name':config.name,'target':d.target,
                'selection_sha256':manifest['selection_sha256'],'objective':asdict(config.objective),
                'margin_weight':config.margin_weight,'kind':'new_appendix_implementation'}
    save_lora_adapter(model,tokenizer,root/'adapter-final',model_config=b.model,lora=b.lora,provenance=provenance)
    write('complete.json',{'kind':'new_appendix_sft_run','metrics':metrics})
    return root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output-dir');p.add_argument('--initial-adapter')
    m=p.add_mutually_exclusive_group(required=True);m.add_argument('--dry-run',action='store_true');m.add_argument('--execute',action='store_true')
    p.add_argument('--model-revision');p.add_argument('--dataset-revision');p.add_argument('--local-data');p.add_argument('--cache-dir');p.add_argument('--allow-download',action='store_true')
    a=p.parse_args(argv)
    try:
        c=load_appendix_sft(a.config);b=c.base
        if a.model_revision:b=replace(b,model=replace(b.model,revision=a.model_revision))
        if a.dataset_revision:b=replace(b,data=replace(b.data,revision=a.dataset_revision))
        c=replace(c,base=b)
        if a.dry_run:
            print(json.dumps(asdict(c),indent=2));print('DRY RUN: no model/data loaded, no files written.')
            return 0
        if not a.output_dir:p.error('--execute requires a new --output-dir')
        result=run_appendix(c,a.output_dir,initial_adapter=a.initial_adapter,allow_download=a.allow_download,local_data=a.local_data,cache_dir=a.cache_dir)
        print(f'Appendix stage complete: {result}')
        return 0
    except (ValueError,OSError,ImportError,RuntimeError) as e:p.exit(2,f'error: {e}\n')
