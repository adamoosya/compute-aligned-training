from pathlib import Path
import csv,json,math,shutil
import numpy as np
import pytest
from cat.paper.reference import (load_sources,curves,reference_audit,corrected_plus_ratio,
                                 diagnostic_rows,sha256,TABLES)
from cat.paper.assets import build_assets
from cat.paper.checks import scan_files,validate_presets
from cat.diagnostics.sensitivity import forced_sensitivities,pivotal_variances

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'results/reference'


def test_reference_sources_and_means():
    a=reference_audit(REFERENCE)
    assert a['sources_verified']==8
    assert a['display_matches']==122 and a['display_comparisons']==124
    mismatches=[x for x in a['checks'] if not x['matches_display']]
    assert len(mismatches)==2 and all(x['group']=='math_rl_passn' for x in mismatches)
    assert not a['standard_errors_recomputed']
    assert a['diagnostics']['majority_diagonal']['count']==81
    assert a['diagnostics']['majority_diagonal']['mean']==pytest.approx(1.2356622617)
    assert a['diagnostics']['bon_diagonal']['count']==277
    assert a['diagnostics']['bon_diagonal']['mean']==pytest.approx(1.2597082916)


@pytest.mark.parametrize('value',[.5,.4,0,1.1,float('nan'),float('inf')])
def test_reject_invalid_plus_ratio(value):
    with pytest.raises(ValueError):corrected_plus_ratio(value)


@pytest.mark.parametrize('value',[.51,.6,.8,.9,1.])
def test_ratio_transform(value):
    assert corrected_plus_ratio(value)==pytest.approx(value/(2*value-1))


def test_modified_reference_rejected(tmp_path):
    d=tmp_path/'ref';shutil.copytree(REFERENCE,d)
    (d/'sources/math_rl_majority.csv').write_text('bad')
    with pytest.raises(ValueError,match='checksum'):load_sources(d)


def test_wrong_sign_convention_rejected(tmp_path):
    d=tmp_path/'ref';shutil.copytree(REFERENCE,d)
    p=d/'manifest.json';m=json.loads(p.read_text());m['sources']['majority_diagonal']['statistic']='a/(a-C)';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='conversion'):load_sources(d)


def test_reference_path_escape_rejected(tmp_path):
    d=tmp_path/'ref';shutil.copytree(REFERENCE,d)
    p=d/'manifest.json';m=json.loads(p.read_text());m['sources']['majority_diagonal']['file']='../outside.csv';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='inside'):load_sources(d)


def test_tables_without_plots_are_means_only_and_create_only(tmp_path):
    before={p:sha256(p) for p in REFERENCE.rglob('*') if p.is_file()}
    result=build_assets(REFERENCE,tmp_path/'out',plots=False)
    assert result['tables']==11 and result['figures']==0
    comp=json.loads((tmp_path/'out/complete.json').read_text())
    assert not comp['standard_errors_recomputed']
    for name,digest in comp['files'].items():assert sha256(tmp_path/'out'/name)==digest
    t=(tmp_path/'out/tables/table4_passn_rl.tex').read_text()
    assert '30.2' in t and '25.2' in t and 'SEs not recomputed' in t
    assert 'TRANSCRIPTION ONLY' in (tmp_path/'out/tables/token_margin_reported.tex').read_text()
    assert before=={p:sha256(p) for p in before}
    with pytest.raises(FileExistsError):build_assets(REFERENCE,tmp_path/'out',plots=False)


def test_raw_curve_budgets():
    _,sources=load_sources(REFERENCE);c=curves(sources)
    assert list(c['math_rl_majority']['Standard_RL'])==[1,4,8,16]
    assert list(c['math_sft_passn']['SFT_Baseline'])==list(range(1,65))
    for _,family,_,budgets,models in TABLES:
        for name,_ in models:assert all(k in c[family][name] for k in budgets)


def test_figure9_and_10_signs_not_mixed():
    _,s=load_sources(REFERENCE)
    for a,b in zip(s['bon_diagonal'],diagnostic_rows(s,'bon_diagonal')):assert float(a['rho'])==b['rho']
    assert all(r['rho']>=1 for r in diagnostic_rows(s,'majority_diagonal'))


def test_single_histogram_renders(tmp_path):
    pytest.importorskip('matplotlib')
    from cat.paper.assets import _histogram
    _,s=load_sources(REFERENCE)
    _histogram(tmp_path,'check',diagnostic_rows(s,'majority_diagonal'),'Majority Vote')
    assert (tmp_path/'check.pdf').read_bytes().startswith(b'%PDF')
    assert (tmp_path/'check.png').read_bytes().startswith(b'\x89PNG')


def test_all_configuration_files_parse():
    assert validate_presets(ROOT)>=40


def test_registry_files_and_scripts_exist():
    registry=json.loads((ROOT/'configs/paper_experiments.json').read_text())
    assert len(registry['experiments'])==9
    for row in registry['experiments']:
        assert (ROOT/'scripts'/row['trainer']).exists()
        assert all((ROOT/x).exists() for x in row['configs'])
        if row['generation_config']:assert (ROOT/row['generation_config']).exists()


def test_credential_scan_hides_value(tmp_path):
    token='hf'+'_'+'a'*35
    (tmp_path/'oops.py').write_text('secret='+repr(token))
    result=scan_files(tmp_path,['oops.py'])
    assert len(result)==1 and 'credential' in result[0]['issue']
    assert token not in json.dumps(result)


def test_machine_path_and_model_archive_scan(tmp_path):
    (tmp_path/'path.py').write_text('path='+repr('/'+'Users/name/private'))
    (tmp_path/'model.pt').write_bytes(b'not a real model')
    r=scan_files(tmp_path,['path.py','model.pt'])
    assert any('personal/cluster' in x['issue'] for x in r)
    assert any('private/model/archive' in x['issue'] for x in r)


def test_new_code_passes_pattern_scan():
    files=[p.relative_to(ROOT).as_posix() for folder in ['src','scripts','configs','docs','results/reference']
           for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert not scan_files(ROOT,files)


@pytest.mark.parametrize('strategy,rewards',[('plurality',None),('best_of_n',[0,2,1])])
def test_diagnostic_repeat_and_bounds(strategy,rewards):
    a=forced_sensitivities([.3,.2,.5],4,strategy=strategy,rewards=rewards,trials=2000)
    b=forced_sensitivities([.3,.2,.5],4,strategy=strategy,rewards=rewards,trials=2000)
    assert a==b
    assert 0<=a['c_rival']<=a['a']+1e-12
    assert a['rho'] is None or a['rho']>=1


def test_n1_diagnostic_exact():
    a=forced_sensitivities([.2,.8],1,trials=100)
    assert a['sensitivities']==[1.,0.]
    assert a['rho']==1


@pytest.mark.parametrize('args',[
    {'probabilities':[.2,.9],'n':4},{'probabilities':[-.1,1.1],'n':4},
    {'probabilities':[0,1],'n':4},{'probabilities':[.2,.8],'n':True},
    {'probabilities':[.2,.8],'n':4,'trials':0},{'probabilities':[.2,.8],'n':4,'target':8},
    {'probabilities':[.2,.8],'n':4,'strategy':'unknown'},
    {'probabilities':[.2,.8],'n':4,'strategy':'best_of_n','rewards':[1,1]}])
def test_diagnostic_validation(args):
    with pytest.raises(ValueError):forced_sensitivities(**args)


@pytest.mark.parametrize('p',[0.,.01,.2,.5,.9,1.])
@pytest.mark.parametrize('n',[2,4,8,16])
def test_rao_blackwell_variance_example(p,n):
    d=pivotal_variances(p,n)
    assert d['conditioned_variance']<=d['sampled_index_variance']+1e-14
    assert d['mean']==pytest.approx(n*p*(1-p)**n)
