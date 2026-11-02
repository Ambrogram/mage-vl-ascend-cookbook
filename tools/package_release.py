"""Deterministic allowlisted ZIP, with an independently verifiable file manifest."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from check_release import ROOT,files,validate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    target=args.out.resolve()
    if target.exists() or target.is_relative_to(ROOT):p.error('Use a fresh ZIP path outside repository')
    report=validate()
    if report['errors']:raise SystemExit(json.dumps(report))
    paths=[x for x in files() if x.name!='RELEASE-MANIFEST.json']
    data={'schema_version':1,'files':{x.relative_to(ROOT).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}
    manifest=ROOT/'RELEASE-MANIFEST.json';manifest.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for x in paths+[manifest]:
            item=zipfile.ZipInfo('mage-vl-ascend-cookbook/'+x.relative_to(ROOT).as_posix(),date_time=(2026,11,2,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED;item.external_attr=0o100644<<16
            z.writestr(item,x.read_bytes())
    with zipfile.ZipFile(target) as z:
        for name,digest in data['files'].items():
            if hashlib.sha256(z.read('mage-vl-ascend-cookbook/'+name)).hexdigest()!=digest:raise RuntimeError('ZIP verification failed')
    print(json.dumps({'zip':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(paths)+1,'all_zip_hashes_verified':True}))

if __name__=='__main__':main()
