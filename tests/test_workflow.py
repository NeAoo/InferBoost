import argparse
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from pipeline.common import digest
from pipeline.models import inspect_model
from pipeline.portable import export_bundle, portable_identity, verify_input
from pipeline.inference.planning import make_trial
from pipeline.quantization_catalog import RECIPES, SCHEMES, eligibility
from pipeline.workflow import build_configs, choose_recipes, resume_contract, verify_resume, reuse_exports


class WorkflowContracts(unittest.TestCase):
    def test_reuse_exports_remeasures_only_complete_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root/'runs/previous'
            candidate = previous/'candidates/w8a8'
            artifact = candidate/'quantized'
            artifact.mkdir(parents=True)
            source = root/'source-model'
            (previous/'config.json').write_text(json.dumps({'source':str(source), 'candidates':[
                {'id':'w8a8','recipe':'compressor_w8a8'}]}))
            (previous/'calibration_manifest.json').write_text(json.dumps({'path':str(previous/'calibration')}))
            (artifact/'quantization_ready.json').write_text('{}')
            (candidate/'model_manifest.json').write_text(json.dumps({'artifact':str(artifact),'mtp_unchanged':True}))
            (candidate/'status.json').write_text('{"status":"failed","error":"audit needs repair"}')
            config = {'source':str(source),'candidates':choose_recipes('w8a8,gptq4')}
            reuse_exports(root,config,'previous')
            self.assertEqual(config['candidates'][0]['mode'],'reuse')
            self.assertEqual(config['candidates'][1]['mode'],'fresh')
            self.assertEqual(config['calibration']['mode'],'frozen')
            self.assertEqual(json.loads((candidate/'status.json').read_text())['status'],'failed')
            config['source']=str(root/'different-source')
            with self.assertRaisesRegex(ValueError,'原模型目录不同'):
                reuse_exports(root,config,'previous')
            config['source']=str(source)
            (artifact/'quantization_ready.json').unlink()
            with self.assertRaisesRegex(ValueError,'完整的导出'):
                reuse_exports(root,config,'previous')

    def test_precision_names_are_distinct_and_hardware_gated(self):
        model = {'model_type': 'qwen3_5', 'adapter': {'family': 'qwen35_dense', 'moe': False}}
        self.assertEqual(SCHEMES['compressor_gptq8'], 'W8A16')
        self.assertEqual(SCHEMES['compressor_w8a8'], 'W8A8')
        self.assertEqual(SCHEMES['compressor_fp8'], 'FP8_DYNAMIC')
        self.assertTrue(eligibility('compressor_w8a8', model, [8, 0])['eligible'])
        self.assertFalse(eligibility('compressor_fp8', model, [8, 0])['eligible'])
        self.assertTrue(eligibility('compressor_fp8', model, [8, 9])['eligible'])
        self.assertNotEqual(RECIPES['compressor_w8a8'].weights, RECIPES['compressor_fp8'].weights)
        for candidate in choose_recipes('all'):
            self.assertEqual(RECIPES[candidate['recipe']].backend, 'llmcompressor')

    def test_new_model_does_not_inherit_private_paths_or_mtp(self):
        root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(model=Path('/models/new'), mtp_model=None, dataset=None,
            recipes='awq4,w8a8', gpus='0,1', max_model_len=4096, max_jobs=2,
            quantization_config=root/'configs/quantization.example.json',
            inference_config=root/'configs/inference.example.json')
        quant, inference = build_configs(root, args)
        self.assertEqual(quant['source'], '/models/new')
        self.assertIsNone(quant['mtp_source'])
        self.assertEqual(inference['search']['mtp_depths'], [0])
        self.assertEqual(quant['runtime']['quantization_gpus'], [0, 1])
        self.assertEqual(inference['runtime']['gpu'], 0)
        self.assertEqual(quant['calibration']['mode'], 'official')
        args.gpus='0,0'
        with self.assertRaises(ValueError): build_configs(root,args)

    def test_portable_model_can_move_but_not_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); model=root/'old'; model.mkdir(); bundle=root/'bundle';bundle.mkdir()
            (model/'config.json').write_text('{"model_type":"qwen3"}')
            (model/'tokenizer.json').write_text('{}')
            header=json.dumps({'layer.weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode()
            (model/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
            identity,_=inspect_model(model,payload_hashes=True)
            trial=make_trial(mtp=0)
            receipt={'schema_version':1,'stage':'portable_deployment','trial':trial,
                     'tested_results':[{'profile':trial['id']}],
                     'identity':portable_identity(identity),'loading_config':identity['config'],
                     'tokenizer_sha256':digest(model/'tokenizer.json')}
            (bundle/'deployment.json').write_text(json.dumps(receipt))
            relocated=root/'new';shutil.copytree(model,relocated)
            verify_input(bundle,relocated)
            with (relocated/'model.safetensors').open('r+b') as stream:
                stream.seek(-1,2);stream.write(b'5')
            with self.assertRaisesRegex(ValueError,'权重'):
                verify_input(bundle,relocated)

    def test_export_rejects_changed_selected_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trial = make_trial()
            (run/'candidates.json').write_text(json.dumps({trial['id']:trial}))
            (run/'selected.json').write_text(json.dumps({'profile':dict(trial,budget=2048)}))
            with patch('pipeline.inference.report.generate', return_value={}):
                with self.assertRaisesRegex(ValueError,'实测候选不一致'):
                    export_bundle(run,run/'deployment')
            self.assertFalse((run/'deployment').exists())

    def test_resume_rejects_changed_data_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); directory=root/'run'; directory.mkdir()
            data=root/'data.json'; data.write_text('[]')
            args=argparse.Namespace(model=root/'model',mtp_model=None,skip_quantization=False)
            quant={'datasets':[{'path':str(data)}],'calibration':{'raw_data':str(data)}}
            inference={'benchmark':{'dataset':str(data)},'quality':{'dataset':str(data)}}
            request={'resume_contract':resume_contract(root,args,quant,inference)}
            (directory/'request.json').write_text(json.dumps(request))
            (directory/'quantization.json').write_text(json.dumps(quant))
            (directory/'inference.json').write_text(json.dumps(inference))
            with patch('pipeline.workflow.build_configs', return_value=(quant,inference)):
                verify_resume(root,directory,args)
                data.write_text('[{}]')
                with self.assertRaisesRegex(ValueError,'已变化'):
                    verify_resume(root,directory,args)
                data.write_text('[]')
                (directory/'inference.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'冻结'):
                    verify_resume(root,directory,args)


if __name__ == '__main__': unittest.main()
