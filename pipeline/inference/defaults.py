"""User-facing defaults shared by the terminal and both report formats."""
from html import escape


DEFAULT_ACCELERATION = (
    ('普通KV Cache', '默认开启', '生成下一个token时复用此前计算过的Key/Value，避免重复计算历史内容。'),
    ('前缀缓存', '默认开启', '跨请求复用相同前缀的计算结果；没有重复前缀时不一定受益。'),
    ('分块预填充', '默认开启', '把长输入拆成多块处理，便于调度其他请求。'),
    ('KV精度', 'auto', '交给运行框架按模型配置决定；它与W/A量化精度是两回事。'),
)
DEFAULTS_NOTE = ('本搜索默认启用这些功能，没有分别做开关消融；实际启动命令中的值保留在工程记录中。'
                 'MTP草稿保持原精度，不参与主模型量化；是否启用MTP解码取决于用户输入和后续测试。')


def defaults_text():
    return ('默认加速设置（开始前先了解）\n' + '\n'.join(
        f'- {name}：{state}。{explanation}' for name, state, explanation in DEFAULT_ACCELERATION)
        + '\n' + DEFAULTS_NOTE + '\n')


def defaults_markdown():
    return ('## 默认加速设置（开始前先了解）\n\n' + '\n'.join(
        f'- **{name}：{state}**。{explanation}' for name, state, explanation in DEFAULT_ACCELERATION)
        + '\n\n' + DEFAULTS_NOTE + '\n\n')


def defaults_html():
    return ('<section id="defaults" class="notice"><h2>默认加速设置（开始前先了解）</h2><dl>'
            + ''.join('<dt>' + escape(name) + '</dt><dd><strong>' + escape(state)
                      + '</strong>。' + escape(explanation) + '</dd>'
                      for name, state, explanation in DEFAULT_ACCELERATION)
            + '</dl><p class="caption">' + escape(DEFAULTS_NOTE) + '</p></section>')
