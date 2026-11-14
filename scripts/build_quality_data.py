"""Create public rule-scored evaluation inputs, independent of calibration data."""
import argparse
import json
from pathlib import Path


def build_rows():
    rows = []
    def add(group, index, prompt, expected, scorer):
        rows.append({'uid': f'quality-{group}-{index:02d}', 'group': group,
                     'messages': [{'role': 'user', 'content': prompt}],
                     'expected': expected, 'scorer': scorer})
    names = ['小林', '小周', '小陈', '小吴']
    for index in range(1, 21):
        start, sent, received = 100 + 7 * index, 12 + index, 3 * index
        add('arithmetic', index,
            f'现有{start}件物品，取走{sent}件，再补入{received}件。现在多少件？只输出最终整数，不要解释或单位。',
            start - sent + received, 'number')
        damaged = index % 2 == 0
        text = '包装已损坏，需要处理。' if damaged else '想了解商品的到货时间。'
        add('classification', index,
            f'严格按规则分类：文字包含“损坏”输出“售后”，否则输出“咨询”。只输出标签。文字：{text} 编号QC{index:02d}。',
            '售后' if damaged else '咨询', 'exact')
        add('extraction', index,
            f'提取信息，只输出JSON对象，键严格为item、quantity、price，不要Markdown或其他字段。内容：购买{index+2}盒彩笔，每盒{index+5}元。',
            {'item': '彩笔', 'quantity': index + 2, 'price': index + 5}, 'json_exact')
        values = [index + 11, index - 8, 2 * index]
        add('json_format', index,
            f'把整数{values}按从小到大排序，只输出JSON数组，不要Markdown和解释。',
            sorted(values), 'json_exact')
        count = 20 if index <= 10 else 100 + 5 * index
        target = (index * 7) % count + 1
        records = '\n'.join(f'编号{number:03d}：负责人{names[(number+index)%4]}，状态已登记。' for number in range(1, count + 1))
        add('reading', index,
            f'以下资料完全虚构。请查询编号{target:03d}的负责人，只输出姓名。\n{records}',
            names[(target + index) % 4], 'exact')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_rows(), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(args.output), 'count': 100,
                      'purpose': 'synthetic_engineering_rule_tests_not_business_benchmark'}))


if __name__ == '__main__':
    main()
