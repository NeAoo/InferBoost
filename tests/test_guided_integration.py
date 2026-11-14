"""CPU integration of HTTP measurements, restart boundaries and guided navigation."""
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pipeline import measurement
from pipeline.common import write_json
from pipeline.quantization_report import generate


class MeasurementHTTPContracts(unittest.TestCase):
    def test_real_sse_three_round_report_and_incomplete_exclusion(self):
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'# synthetic metrics\n')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                if self.path == '/reset_prefix_cache':
                    self.end_headers()
                    self.wfile.write(b'true')
                    return
                bodies.append(body)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for content, finish in [('O', None), ('K', 'stop')]:
                    event = {'choices': [{'delta': {'content': content}, 'finish_reason': finish}]}
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                    self.wfile.flush()
                    time.sleep(.002)
                usage = {'usage': {'prompt_tokens': 3, 'completion_tokens': body['max_tokens'] if body.get('ignore_eos') else 2}}
                self.wfile.write(('data: ' + json.dumps(usage) + '\n\ndata: [DONE]\n\n').encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{'uid': name, 'group': 'same', 'messages': [{'role': 'user', 'content': name}],
                     'expected': 'OK', 'scorer': 'exact'} for name in ('a', 'b')]
            settings = {'performance_repeats': 3, 'separate_quality': True, 'concurrencies': [1],
                        'cache_modes': ['cold', 'warm'], 'max_tokens': 32, 'timeout_seconds': 5}
            write_json(root / 'config.json', {'candidates': [{'id': 'original', 'recipe': 'original'},
                       {'id': 'gptq4', 'recipe': 'compressor_gptq4'}], 'datasets': [{'id': 'test'}],
                       'benchmark': settings, 'quality_gate': {'max_drop_pp': 0}, 'selection': {'objective': 'latency'}})
            write_json(root / 'datasets/test.json', rows)
            for candidate in ('original', 'gptq4'):
                directory = root / 'candidates' / candidate
                measurement.measure(f'http://127.0.0.1:{server.server_port}', rows, settings,
                                    directory / 'datasets/test', candidate)
                write_json(directory / 'status.json', {'status': 'completed'})
            self.assertEqual(len(bodies), 40)  # 2 candidates × (2 quality + 12 performance + 6 warmup)
            self.assertTrue(all(body['seed'] == 0 and body['temperature'] == 0 for body in bodies))
            result = generate(root)
            self.assertTrue(all(result['selectable'].values()))
            self.assertEqual(result['overview'][0]['requests'], 2)
            self.assertEqual(result['overview'][0]['score_pct'], 100)
            self.assertEqual(result['overview'][0]['performance_repeats'], 3)
            self.assertIsNotNone(result['overview'][0]['e2e_s_std'])
            broken = root / 'candidates/gptq4/datasets/test/performance/round-3/requests.jsonl'
            broken.write_text('\n'.join(broken.read_text().splitlines()[:-1]) + '\n')
            result = generate(root)
            self.assertFalse(result['selectable']['gptq4'])
            self.assertFalse(result['candidate_gates']['gptq4']['passed'])
            self.assertEqual(result['recommendation']['candidate'], 'original')


class WorkspaceEntryContracts(unittest.TestCase):
    def test_first_entry_remembers_workspace_and_preserves_results(self):
        try:
            from pipeline import ui
        except ImportError:
            self.skipTest('terminal dependencies not installed')
        from image.entry import enter
        from image.workspace import MANIFEST, fingerprints
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            source, destination, home = root / 'source', root / 'volume', root / 'home'
            source.mkdir()
            (source / 'quickstart').write_text('entry')
            write_json(source / MANIFEST, fingerprints(source))
            stack.enter_context(patch('pathlib.Path.home', return_value=home))
            stack.enter_context(patch.dict(os.environ, {'PIPELINE_WORKSPACE': '/workspace/inferboost-v050', 'TERM': 'xterm-256color'}))
            stack.enter_context(patch('sys.stdin.isatty', return_value=True))
            prompt = stack.enter_context(patch.object(ui, 'text_input', return_value=str(destination)))
            stack.enter_context(patch.object(ui, 'choose', side_effect=['path', 'use']))
            execute = stack.enter_context(patch('os.execv'))
            enter(source, [])
            self.assertEqual(prompt.call_count, 1)
            self.assertEqual(execute.call_args.args[0], str(destination.resolve() / 'quickstart'))
            write_json(destination / 'runs/result.json', {'kept': True})
            os.environ['PIPELINE_WORKSPACE'] = '/workspace/inferboost-v050'
            enter(source, [])
            self.assertEqual(prompt.call_count, 1)
            self.assertTrue((destination / 'runs/result.json').is_file())


class GuidedNavigationContracts(unittest.TestCase):
    def test_back_preserves_model_and_cancel_submits_nothing(self):
        try:
            from pipeline import guided, ui
        except ImportError:
            self.skipTest('terminal dependencies not installed')
        resources = {'storage': {'path': '/work', 'free_gib': 100, 'persistence': '未知'},
                     'memory': {'limit_gib': None}, 'gpus': []}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(guided.preflight, 'inspect_resources', return_value=resources))
            stack.enter_context(patch.object(guided, 'model_input', side_effect=lambda value, skip: Path(value)))
            stack.enter_context(patch.object(guided.preflight, 'model_summary', return_value={
                'model_type': 'qwen3', 'precision': 'bf16', 'weight_gib': 1, 'shards': 1,
                'quantized': False, 'mtp_tensors': 0, 'max_model_len': 1024}))
            prompt = stack.enter_context(patch.object(ui, 'text_input', side_effect=['/model', ui.Back(), ui.Cancel()]))
            submit = stack.enter_context(patch.object(guided.jobs, 'submit'))
            with self.assertRaises(ui.Cancel):
                guided.guide(Path(directory))
            self.assertEqual(prompt.call_args_list[-1].args[:2], ('模型目录', '/model'))
            submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
