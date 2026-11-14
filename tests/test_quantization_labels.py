import unittest

from pipeline.quantization_labels import describe_quantization, quantization_sort_key


class QuantizationLabels(unittest.TestCase):
    def test_algorithm_and_activation_format_are_independent(self):
        expected = {'compressor_awq4': 'AWQ-W4A16', 'compressor_gptq4': 'GPTQ-W4A16',
                    'compressor_awq_gptq4': 'AWQ＋GPTQ-W4A16', 'compressor_gptq8': 'GPTQ-W8A16',
                    'compressor_w8a8': 'GPTQ-W8A8', 'compressor_w4afp8': 'AWQ-W4AFP8',
                    'compressor_fp8': 'RTN-W8A8-FP8'}
        for recipe, label in expected.items():
            with self.subTest(recipe=recipe):
                self.assertEqual(describe_quantization(recipe)['quantization_label'], label)
        self.assertEqual(describe_quantization('compressor_gptq8')['weight_precision'],
                         describe_quantization('compressor_w8a8')['weight_precision'])
        self.assertIn('BF16', describe_quantization('compressor_gptq8')['activation_precision'])
        self.assertIn('动态', describe_quantization('compressor_w8a8')['activation_precision'])

    def test_external_model_format_does_not_prove_quantization_algorithm(self):
        unknown = describe_quantization(precision='W8A8')
        self.assertEqual(unknown['algorithm'], '未记录')
        self.assertEqual(unknown['quantization_label'], '算法未记录 · W8A8')
        self.assertEqual(describe_quantization('original')['activation_precision'], '保持源模型精度')

    def test_sort_groups_algorithm_before_precision_without_changing_ids(self):
        recipes = ['compressor_awq_gptq4', 'compressor_w8a8', 'compressor_awq4',
                   'compressor_gptq8', 'compressor_gptq4', 'original']
        rows = [{'recipe': recipe, 'candidate': str(index)} for index, recipe in enumerate(recipes)]
        ordered = sorted(rows, key=quantization_sort_key)
        self.assertEqual([row['recipe'] for row in ordered],
                         ['original', 'compressor_gptq4', 'compressor_gptq8', 'compressor_w8a8',
                          'compressor_awq4', 'compressor_awq_gptq4'])
        self.assertEqual([row['candidate'] for row in ordered], ['5', '4', '3', '1', '2', '0'])


if __name__ == '__main__':
    unittest.main()
