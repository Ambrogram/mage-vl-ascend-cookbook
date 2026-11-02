import ast,contextlib,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mage_ascend.cli import verify,write,qualification,parser
from mage_ascend import model_runtime as m,gate_adapter

class Contracts(unittest.TestCase):
    def test_manifest_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'config').write_bytes(b'original')
            f=r/'manifest.json';f.write_text(json.dumps({'config':hashlib.sha256(b'original').hexdigest()}))
            self.assertEqual(verify(r,f)['files_verified'],1)
            (r/'config').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):verify(r,f)
    def test_manifest_rejects_escape(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);f=r/'manifest.json'
            for name in ['../secret','/etc/secret','a\\secret','C:secret']:
                f.write_text(json.dumps({name:'0'*64}))
                with self.assertRaises(ValueError):verify(r,f)
    def test_no_result_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json';write(path,{'status':'unknown'})
            with self.assertRaises(FileExistsError):write(path,{'status':'passed'})
            self.assertEqual(json.loads(path.read_text())['status'],'unknown')
    def test_gate_failed_receipt_cannot_enable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json';write(p,{'status':'failed'})
            with self.assertRaises(ValueError):qualification(p,'npu:0')
    def test_generation_distinguishes_truncation(self):
        events=[{'event':'pre'},{'event':'post'}]
        result=m.generation_evidence([3,4],[9],2,events)
        self.assertFalse(result['ended_in_eos']);self.assertEqual(result['stop_reason'],'max_new_tokens')
        self.assertTrue(m.generation_evidence([3,9],[9],2,events)['ended_in_eos'])
    def test_no_visual_forward_not_accepted(self):
        self.assertFalse(m.generation_evidence([9],[9],2,[])['visual_contract_valid'])
    def test_mamba_pinned_sources(self):
        for name,digest in gate_adapter.UPSTREAM_SHA.items():
            self.assertEqual(hashlib.sha256((ROOT/'src/mage_ascend/gate-upstream'/name).read_bytes()).hexdigest(),digest)
    def test_help_needs_no_accelerator(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out),self.assertRaises(SystemExit) as ctx:parser().parse_args(['--help'])
        self.assertEqual(ctx.exception.code,0);self.assertIn('doctor',out.getvalue())
    def test_stale_gate_receipt_cannot_enable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            data={'status':'passed','device':'npu:0','gate_dtype':'float32','prefixes':[16,32,48,64],
                  'adapter_sha256':'0'*64,'gate_weight_sha256':gate_adapter.GATE_WEIGHT_SHA}
            write(p,data)
            with self.assertRaises(ValueError):qualification(p,'npu:0')
    def test_wrong_device_gate_receipt_cannot_enable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            write(p,{'status':'passed','device':'npu:1','gate_dtype':'float32','prefixes':[16,32,48,64],
                'adapter_sha256':hashlib.sha256(Path(gate_adapter.__file__).read_bytes()).hexdigest(),
                'gate_weight_sha256':gate_adapter.GATE_WEIGHT_SHA})
            with self.assertRaises(ValueError):qualification(p,'npu:0')

if __name__=='__main__':unittest.main()
