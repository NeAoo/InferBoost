import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from pipeline.benchmark import read_sse, request_one, percentile
from pipeline.common import identifier, inside
from pipeline.models import inspect_model
from pipeline.cli import select, signature


class PipelineContracts(unittest.TestCase):
    def test_path_escape(self):
        with self.assertRaises(ValueError): identifier('../old')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): inside(directory,Path(directory)/'../other')

    def test_sse_multiline_and_done(self):
        lines=[b': heartbeat\n',b'\n',b'data: {"a":\n',b'data: 1}\n',b'\n',b'data: [DONE]\n',b'\n']
        self.assertEqual(list(read_sse(lines)),['{"a":\n1}','[DONE]'])

    def test_truncated_shard_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            header=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
            shard=root/'model.safetensors';shard.write_bytes(struct.pack('<Q',len(header))+header+b'\0\0')
            with self.assertRaises(ValueError): inspect_model(root)
            shard.write_bytes(struct.pack('<Q',len(header))+header+b'\0\0\0\0')
            info,_=inspect_model(root,payload_hashes=True)
            self.assertEqual(info['tensor_count'],1)

    def test_sse_missing_done_is_failure(self):
        class Response:
            headers=type('Headers',(),{'get_content_type':lambda self:'text/event-stream'})()
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                event={'choices':[{'delta':{'content':'hello'},'finish_reason':'stop'}],
                       'usage':{'prompt_tokens':2,'completion_tokens':1}}
                return iter([('data: '+json.dumps(event)+'\n').encode(),b'\n'])
        opener=type('Opener',(),{'open':lambda self,*args,**kwargs:Response()})()
        with patch('urllib.request.build_opener',return_value=opener):
            result=request_one('http://localhost',{'uid':'1','group':'test','messages':[{'role':'user','content':'x'}]}, {'max_tokens':16})
        self.assertEqual(result['status'],'error')
        self.assertFalse(result['done'])

    def test_failed_profile_cannot_be_deployed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/r1';(run/'profiles/p1').mkdir(parents=True)
            (run/'config.json').write_text(json.dumps({'profiles':[{'id':'p1'}]}))
            (run/'profiles/p1/status.json').write_text('{"status":"failed"}')
            with self.assertRaises(ValueError): select(root,run,'p1')
            self.assertFalse((run/'selected.json').exists())

    def test_single_stream_chunk_has_no_observable_decode_interval(self):
        class Response:
            headers=type('Headers',(),{'get_content_type':lambda self:'text/event-stream'})()
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self):
                event={'choices':[{'delta':{'content':'one combined chunk'},'finish_reason':'stop'}],
                       'usage':{'prompt_tokens':2,'completion_tokens':8}}
                return iter([('data: '+json.dumps(event)+'\n').encode(),b'\n',b'data: [DONE]\n',b'\n'])
        opener=type('Opener',(),{'open':lambda self,*args,**kwargs:Response()})()
        with patch('urllib.request.build_opener',return_value=opener):
            result=request_one('http://localhost',{'uid':'1','group':'test','messages':[{'role':'user','content':'x'}]}, {'max_tokens':16})
        self.assertEqual(result['status'],'success')
        self.assertIsNone(result['decode_tps'])
        self.assertIsNone(result['tpot_ms'])

    def test_percentile_direction(self):
        self.assertAlmostEqual(percentile([1,2,3,4,5],.95),4.8)
        self.assertIsNone(percentile([None],.95))

    def test_resume_signature_changes_with_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data.json').write_text('[]')
            (root/'model').mkdir();(root/'model/config.json').write_text('{}')
            config={'benchmark':{'dataset':str(root/'data.json')},'quantization':{'model':str(root/'model')}}
            before=signature(root,config);(root/'data.json').write_text('[1]')
            self.assertNotEqual(before,signature(root,config))


if __name__=='__main__': unittest.main()
