"""Explicit offline-model CLI. No automatic installation, download, or retry."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def verify(model_dir, manifest):
    """Verify caller-selected reference before executing remote model code."""
    root=Path(model_dir).resolve(strict=True)
    values=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if not isinstance(values,dict) or not values: raise ValueError('Empty/invalid manifest')
    result={}
    for name, expected in values.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Manifest path escapes model root')
        p=root/relative
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file():
            raise ValueError('Manifest entry is absent, symlinked or outside model root')
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Invalid digest')
        result[name]=sha(p)
        if result[name]!=expected: raise ValueError('Model identity mismatch: '+name)
    return {'status':'passed','files_verified':len(result),'manifest_sha256':sha(manifest)}


def write(path, data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,default=str)


def settings(args):
    return SimpleNamespace(num_frames=16,max_pixels=150000,target_fps=None,frame_cap=16,
        max_input_tokens=8192,max_new_tokens=args.max_new_tokens,probe_only=False,
        min_free_before_generate_gib=4.0)


def qualification(path, device):
    from . import gate_adapter
    r=json.loads(Path(path).read_text(encoding='utf-8'))
    if (r.get('status')!='passed' or r.get('device')!=device or r.get('gate_dtype')!='float32'
        or r.get('prefixes')!=[16,32,48,64]
        or r.get('adapter_sha256')!=sha(gate_adapter.__file__)
        or r.get('gate_weight_sha256')!=gate_adapter.GATE_WEIGHT_SHA):
        raise ValueError('Requires independent full-gate oracle and prefix qualification on target device')
    return r


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    v=subs.add_parser('verify'); v.add_argument('--model-dir',required=True);v.add_argument('--manifest',required=True);v.add_argument('--out',required=True)
    d=subs.add_parser('doctor');d.add_argument('--device',required=True);d.add_argument('--out',required=True)
    i=subs.add_parser('infer')
    for flag in ['model-dir','manifest','input','out']: i.add_argument('--'+flag,required=True)
    i.add_argument('--mode',choices=['image','video','stream_fast'],required=True)
    i.add_argument('--device',required=True)
    i.add_argument('--prompt',default='Describe the main visible events in chronological order.')
    i.add_argument('--max-new-tokens',type=int,default=128)
    i.add_argument('--gate-qualification')
    return p


def infer(args):
    from . import model_runtime as m, video_inputs as v, stream_fast as fast
    verified=verify(args.model_dir,args.manifest)
    folder=Path(args.out).resolve();folder.mkdir(parents=True,exist_ok=False)
    evidence={'status':'prepared','model_verification':verified,'mode':args.mode,
              'public_runtime_npu_revalidation':'not implied by historical evidence',
              'automatic_retry':False,'input_sha256':sha(args.input)}
    def persist(): m.save_json(folder/'result.json',evidence)
    persist()
    try:
        m.ROOT=folder;m.MODEL_DIR=Path(args.model_dir).resolve()
        v.EXPERIMENT_ROOT=folder;v.MODEL_DIR=m.MODEL_DIR;fast.ROOT=folder
        if args.mode=='stream_fast': qualification(args.gate_qualification,args.device)
        runtime=m.Runtime(SimpleNamespace(instance_id=uuid.uuid4().hex,device=args.device,
                                         dtype='bfloat16',attention='sdpa'))
        runtime.load()
        if not runtime.snapshot()['ready']: raise RuntimeError('Load failed; inspect instance load.json')
        if args.mode=='image':
            runtime.infer(Path(args.input).read_bytes(),args.prompt,args.max_new_tokens,evidence)
            evidence['status']='inference_completed_semantic_review_pending'
        elif args.mode=='video':
            v.run_video_case(runtime,runtime.processor,Path(args.input),args.prompt,settings(args),
                             evidence=evidence,persist=persist,helpers=m)
        else:
            import torch
            from . import gate_adapter
            gate,diagnostic=gate_adapter.load_gate(m.MODEL_DIR,runtime.device,torch.float32,
                                                  scan_backend='npu_reference',gate_attention='sdpa')
            evidence['gate_load']=diagnostic
            fast.stream_replay(runtime,v,m,args.input,args.prompt,settings(args),evidence,persist,folder,gate=gate)
        persist()
        print(json.dumps({'out':str(folder),'status':evidence['status']},ensure_ascii=False))
    except BaseException as exc:
        evidence.update(status='failed',error_type=type(exc).__name__)
        persist()
        raise


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command=='verify':write(args.out,verify(args.model_dir,args.manifest));return 0
    if not re.fullmatch(r'npu:[0-9]+',args.device):p.error('Explicit logical npu:N required')
    if args.command=='doctor':
        import platform,torch,torch_npu,transformers
        torch.npu.set_device(args.device)
        if not torch.npu.is_available(): raise RuntimeError('NPU not available')
        x=torch.tensor([1.,2.],device=args.device);y=x.square();torch.npu.synchronize()
        if y.cpu().tolist()!=[1.,4.]:raise RuntimeError('NPU compute probe failed')
        write(args.out,{'python':platform.python_version(),'machine':platform.machine(),'device':args.device,
            'torch':torch.__version__,'torch_npu':torch_npu.__version__,'transformers':transformers.__version__,
            'visible_devices':os.getenv('ASCEND_RT_VISIBLE_DEVICES'),'compute_probe':'passed',
            'free_total_bytes':list(torch.npu.mem_get_info(args.device))});return 0
    if not 1<=args.max_new_tokens<=128 or not args.prompt.strip() or len(args.prompt)>4096:
        p.error('Budget is 1..128 tokens; prompt must be 1..4096 nonblank characters')
    if args.mode=='stream_fast' and not args.gate_qualification:p.error('stream_fast requires --gate-qualification')
    infer(args);return 0


if __name__=='__main__':raise SystemExit(main())
