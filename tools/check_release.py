"""Offline publication checks. Never loads Torch, a model, or network credentials."""
from pathlib import Path
import ast,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.py','.md','.json','.toml','.txt','.yml','.gitignore'}
EXCLUDED={'.git','__pycache__','runs','dist','build','.venv'}
def files(root=ROOT):
    for p in sorted(root.rglob('*')):
        relative=p.relative_to(root)
        if any(x in EXCLUDED or x.endswith('.egg-info') for x in relative.parts):continue
        if p.is_symlink():raise ValueError('Symlink not publishable: '+str(relative))
        if p.is_file():yield p

def validate(root=ROOT):
    errors=[];count=0
    for p in files(root):
        name=p.relative_to(root).as_posix();count+=1
        if p.name not in {'LICENSE','NOTICE','.gitignore','Microsoft-Mage-LICENSE'} and p.suffix not in ALLOWED:
            errors.append('Unapproved type: '+name)
        if p.stat().st_size>2*1024*1024:errors.append('Oversized: '+name)
        try:s=p.read_text(encoding='utf-8')
        except UnicodeError:errors.append('Binary/non UTF-8: '+name);continue
        # Patterns are assembled to avoid self-triggering in the checker source.
        needles=['C:'+chr(92)+'Users'+chr(92), 'D:'+chr(92)+'WorkProject',
                 'ailab.'+'sinomach', 'ai.'+'sinomach.com.cn', '/home/bml/'+'storage']
        if any(n.lower() in s.lower() for n in needles):errors.append('Private location: '+name)
        if re.search(r'(?i)(?:cookie|authorization)\s*[:=]\s*[\x22\x27]?(?:Bearer\s+)?[A-Za-z0-9_\-+/=]{32,}',s):
            errors.append('Possible secret: '+name)
        if p.suffix=='.py':
            try:ast.parse(s,filename=name)
            except SyntaxError:errors.append('Syntax: '+name)
        if p.suffix=='.json':
            try:json.loads(s,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
            except ValueError:errors.append('Non-strict JSON: '+name)
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',s):
                target=link.split('#',1)[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (p.parent/target).resolve().is_relative_to(root.resolve()) or not (p.parent/target).exists():
                    errors.append('Broken local link: '+name+' -> '+target)
    prov=root/'evidence/code-provenance.json'
    if prov.exists():
        for row in json.loads(prov.read_text(encoding='utf-8')):
            path=root/row['published_path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['published_sha256']:
                errors.append('Provenance mismatch: '+row['published_path'])
    return {'status':'failed' if errors else 'passed','files_checked':count,'errors':errors,
            'scope':'UTF-8/type/path/secret heuristics, syntax, JSON, local links and derivation hashes; not NPU acceptance'}

if __name__=='__main__':
    result=validate();print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(bool(result['errors']))
