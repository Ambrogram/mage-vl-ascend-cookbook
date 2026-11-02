"""Export real 64-frame native vision features without loading a gate.

Qualification fixture only, not a streaming or latency benchmark. A 64-frame
visual pass can require substantially more memory than the 16-frame service.
"""
import argparse,uuid
from pathlib import Path
from types import SimpleNamespace
from mage_ascend.cli import verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir','manifest','video','device','out']:p.add_argument('--'+name,required=True)
    args=p.parse_args()
    verify(args.model_dir,args.manifest)
    from mage_ascend import model_runtime as m,video_inputs as v
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    m.ROOT=root;m.MODEL_DIR=Path(args.model_dir).resolve();v.EXPERIMENT_ROOT=root
    r=m.Runtime(SimpleNamespace(instance_id=uuid.uuid4().hex,device=args.device,dtype='bfloat16',attention='sdpa'))
    r.load()
    if not r.snapshot()['ready']:raise RuntimeError('Load failed')
    inputs,evidence=v.build_video_inputs(r.processor,args.video,'Describe the video.',64,150000,frame_cap=64)
    if evidence['decoded_frame_count']!=64:raise ValueError('Video must provide 64 genuine frames')
    device_inputs={k:t.to(r.device) for k,t in inputs.items()}
    device_inputs['pixel_values']=device_inputs['pixel_values'].to(r.dtype)
    v.export_gate_features(r,device_inputs,root/'features.safetensors',evidence,m.EXPECTED_SOURCES)
    print(root/'features.safetensors')

if __name__=='__main__':main()
