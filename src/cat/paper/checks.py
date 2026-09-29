"""Local release preflight. Does not stage, commit, push or change files."""
from __future__ import annotations
import argparse,ast,json,re,subprocess
from pathlib import Path
from cat.paper.reference import read_json,reference_audit


def validate_presets(root):
    from cat.experiments.sft_config import load_sft_config
    from cat.experiments.rl_config import load_rl_config
    from cat.experiments.generation_config import load_generation_config
    from cat.experiments.appendix_sft import load_appendix_sft
    from cat.evaluation.scoring import load_score_config
    root=Path(root); count=0
    for path in sorted((root/'configs').rglob('*.json')):
        if path.name=='paper_experiments.json':continue
        value=read_json(path)
        if 'base' in value:load_appendix_sft(path)
        elif 'weights' in value:load_rl_config(path)
        elif 'scoring' in value:load_generation_config(path)
        elif 'budgets' in value:load_score_config(path)
        else:load_sft_config(path)
        count+=1
    return count


def scan_files(root,paths):
    root=Path(root);issues=[]
    secrets=[re.compile(r'\b'+'hf'+r'_[A-Za-z0-9]{24,}\b'),
             re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
             re.compile(r'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
             re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    machine_paths=[re.compile('/'+r'ocean/projects/[^\s\"\']+'),re.compile('/'+r'jet/home/[^\s\"\']+'),
                   re.compile('/'+r'Users/[^\s\"\']+')]
    prohibited={'.pt','.pth','.safetensors','.bin','.tar','.gz','.zip','.pem','.key'}
    for name in sorted(set(paths)):
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            issues.append({'file':name,'issue':'unsafe path'});continue
        path=root/rel
        if any((root/p).is_symlink() for p in (rel,*rel.parents)):
            issues.append({'file':name,'issue':'symlink must be reviewed'});continue
        if not path.is_file():continue # A staged deletion is not published from this worktree.
        if path.suffix in prohibited or path.name=='.env' or path.name.startswith('.env.'):
            if path.name!='.env.example':issues.append({'file':name,'issue':'private/model/archive file in release set'})
        if path.stat().st_size>10*1024*1024:
            issues.append({'file':name,'issue':'file exceeds 10 MiB; review before publishing'});continue
        if path.suffix.lower() in {'.png','.pdf','.jpg','.jpeg','.webp'}:continue
        try:text=path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            issues.append({'file':name,'issue':'unexpected binary file'});continue
        for line_no,line in enumerate(text.splitlines(),1):
            if any(p.search(line) for p in secrets):
                issues.append({'file':name,'line':line_no,'issue':'possible credential (value not displayed)'})
            if any(p.search(line) for p in machine_paths):
                issues.append({'file':name,'line':line_no,'issue':'hard-coded personal/cluster path'})
        if path.suffix=='.py':
            try:ast.parse(text,filename=name)
            except SyntaxError as e:issues.append({'file':name,'line':e.lineno,'issue':'Python syntax error'})
    return issues


def check_release(root):
    root=Path(root).resolve()
    proc=subprocess.run(['git','-C',str(root),'rev-parse','--show-toplevel'],capture_output=True,text=True)
    if proc.returncode or Path(proc.stdout.strip()).resolve()!=root:
        raise ValueError('Run from the repository root')
    proc=subprocess.run(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True)
    paths=[s.decode('utf-8') for s in proc.stdout.split(b'\0') if s]
    issues=scan_files(root,paths)
    required=['README.md','CITATION.cff','pyproject.toml','docs/experiments.md','docs/reproducibility.md',
              'docs/release.md','results/reference/manifest.json','.github/workflows/tests.yml']
    for name in required:
        if not (root/name).is_file():issues.append({'file':name,'issue':'required release file missing'})
    for directory in ['outputs','checkpoints']:
        p=subprocess.run(['git','-C',str(root),'check-ignore',directory+'/probe.bin'],capture_output=True)
        if p.returncode:issues.append({'file':'.gitignore','issue':f'{directory}/ is not excluded'})
    audit=reference_audit(root/'results/reference')
    configs=validate_presets(root)
    # Data discrepancies remain explicit; two known display differences are not changed.
    if audit['display_matches']!=122 or audit['display_comparisons']!=124:
        issues.append({'file':'results/reference','issue':'unexpected reference comparison change'})
    warnings=[]
    if not any((root/x).is_file() for x in ['LICENSE','LICENSE.txt','LICENSE.md']):
        warnings.append('Code license has not been selected; add the approved LICENSE before public release.')
    citation=(root/'CITATION.cff').read_text(encoding='utf-8') if (root/'CITATION.cff').is_file() else ''
    if 'repository-code:' not in citation:
        warnings.append('Public repository URL is not set in CITATION.cff; add it after choosing the GitHub destination.')
    warnings.append('CUDA/NF4 and full pretrained training reproduction are not validated by CPU tests.')
    return {'files_scanned':len(set(paths)),'configurations_validated':configs,'reference_sources_verified':audit['sources_verified'],
            'issues':issues,'warnings':warnings,'scope':'Worktree candidate-file preflight; limited pattern scan, not an exhaustive credential or Git-history audit.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--strict',action='store_true')
    args=p.parse_args(argv)
    try:report=check_release(args.repo)
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as e:p.exit(2,f'error: {e}\n')
    print(json.dumps(report,indent=2))
    if report['issues']:print('Release preflight found files to review.');return 1
    if args.strict and any('license' in w or 'URL' in w for w in report['warnings']):
        print('Code checks passed; publication metadata still needs confirmation.');return 1
    print('Release code/assets checks passed. Publication decisions are listed above.')
    return 0
