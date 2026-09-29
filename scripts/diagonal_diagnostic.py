#!/usr/bin/env python3
"""Estimate sensitivities from a user-supplied categorical probability vector."""
import argparse,json
from pathlib import Path
from cat.diagnostics.sensitivity import forced_sensitivities
from cat.paper.reference import read_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='JSON with probabilities and optional rewards/target')
    p.add_argument('--strategy',choices=['plurality','best_of_n'],default='plurality')
    p.add_argument('--n',type=int,default=8);p.add_argument('--trials',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.is_symlink():p.error('Output must be new')
    try:
        value=read_json(args.input)
        if set(value)-{'probabilities','rewards','target'}:raise ValueError('Unknown input fields')
        result=forced_sensitivities(value['probabilities'],args.n,target=value.get('target',0),
            strategy=args.strategy,rewards=value.get('rewards'),trials=args.trials,seed=args.seed)
        with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    except (ValueError,KeyError,OSError) as e:p.exit(2,f'error: {e}\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
