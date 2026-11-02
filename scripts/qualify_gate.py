"""Independent full-gate FP32 comparison on a 64-frame real vision feature fixture.

Run on a free target NPU before stream_fast. Both gates load sequentially, never
simultaneously. Not an automatic approval based on a tiny/random scan test.
"""
import argparse,gc,json
from pathlib import Path
from mage_ascend import gate_adapter as a
from mage_ascend.cli import sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--features',required=True)
    p.add_argument('--device',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    import torch,torch_npu
    from safetensors.torch import load_file
    if Path(args.out).exists():p.error('Output exists')
    device=torch.device(args.device);torch.npu.set_device(device)
    features=load_file(args.features)['vision_tokens']
    if features.ndim!=4 or features.shape[0]!=1 or features.shape[1]!=64 or features.shape[-1]!=2560:
        p.error('Requires real [1,64,patch_tokens,2560] gate vision features')
    x=features.to(device=device,dtype=torch.float32)
    results={}
    with torch.inference_mode():
        for backend in ['official_reference','npu_reference']:
            gate,_=a.load_gate(args.model_dir,device,torch.float32,scan_backend=backend,gate_attention='sdpa')
            values={}
            for count in [16,32,48,64]:
                logits=gate(x[:,:count], response_positions=list(range(16,count+1,16)))[0].detach().float().cpu()
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite gate output')
                values[count]=logits
            results[backend]=values
            del gate;gc.collect();torch.npu.empty_cache()
    failures=[];maxdiff=0.
    for n in [16,32,48,64]:
        ref,candidate=results['official_reference'][n],results['npu_reference'][n]
        maxdiff=max(maxdiff,float((ref-candidate).abs().max()))
        for lhs,rhs in [(ref,candidate),(candidate,results['npu_reference'][64][:n])]:
            if not torch.allclose(lhs,rhs,rtol=2e-3,atol=2e-3):failures.append('logits_'+str(n))
            if not torch.equal(lhs.softmax(-1)[:,1]>=.5,rhs.softmax(-1)[:,1]>=.5):failures.append('decisions_'+str(n))
    write(args.out,{'status':'failed' if failures else 'passed','failures':failures,
        'device':str(device),'gate_dtype':'float32','prefixes':[16,32,48,64],
        'adapter_sha256':sha(a.__file__),'gate_weight_sha256':a.GATE_WEIGHT_SHA,
        'features_sha256':sha(args.features),'max_abs_oracle_error':maxdiff,
        'rtol':2e-3,'atol':2e-3,'semantic_quality':'not assessed'})
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
