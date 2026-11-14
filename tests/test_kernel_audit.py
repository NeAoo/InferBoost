from types import SimpleNamespace
import unittest

from pipeline.kernel_audit import loaded_quantization


class LoadedKernelContracts(unittest.TestCase):
    def test_compressed_tensors_scheme_lives_on_layer(self):
        kernel = object()
        scheme = SimpleNamespace(kernel=kernel)
        method = SimpleNamespace(quantization_config={})
        layer = SimpleNamespace(quant_method=method, scheme=scheme)
        actual_method, actual_scheme, actual_kernel = loaded_quantization(layer)
        self.assertIs(actual_method, method)
        self.assertIs(actual_scheme, scheme)
        self.assertIs(actual_kernel, kernel)

    def test_direct_kernel_and_wrapped_methods_remain_visible(self):
        kernel = object()
        method = SimpleNamespace(kernel=kernel)
        self.assertEqual(loaded_quantization(SimpleNamespace(base_quant_method=method)),
                         (method, None, kernel))
        self.assertEqual(loaded_quantization(SimpleNamespace()), (None, None, None))
        scheme = SimpleNamespace(kernel=kernel)
        method = SimpleNamespace(scheme=scheme)
        self.assertEqual(loaded_quantization(SimpleNamespace(quant_method=method)),
                         (method, scheme, kernel))
