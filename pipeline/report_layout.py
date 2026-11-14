"""Small offline HTML primitives shared by the two public reports."""
import html


STYLE = '''
:root{color-scheme:light;--ink:#202b30;--muted:#53636b;--line:#dce3e5;--soft:#f4f7f7;--accent:#075d53;--selected:#edf6f3}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.75 system-ui,-apple-system,sans-serif}
main{max-width:1160px;margin:36px auto 72px;padding:0 28px}h1{font-size:28px;line-height:1.4;margin:0 0 12px}h2{font-size:21px;line-height:1.5;margin:36px 0 10px}h3{font-size:16px;margin:20px 0 8px}p{margin:10px 0;max-width:85ch}a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:4px}::selection{background:#cce9df;color:#173d34}
.meta,small{color:var(--muted)}.lead{font-size:17px}.notice{background:var(--soft);padding:14px 18px;margin:18px 0}.notice p{margin:0}.table-wrap{max-width:100%;overflow-x:auto;scrollbar-color:#8eaba3 var(--soft)}
table{width:100%;border-collapse:collapse;font-size:14px;margin:14px 0 20px;font-variant-numeric:tabular-nums}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:var(--soft);font-weight:600}td:first-child,th:first-child{text-align:left}.winner{background:var(--selected)}.winner td:first-child{font-weight:650}.choice{color:var(--accent);font-weight:650}.tag{display:block;font-size:12px;line-height:1.5;color:var(--accent)}.state-note{display:block;font-size:12px;line-height:1.5;color:var(--muted)}th.cache-group{text-align:center}.group-start{border-left:1px solid var(--line)}
details{margin:16px 0;border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-weight:600;min-height:30px}pre{padding:14px;background:var(--soft);white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}code{overflow-wrap:anywhere}dl{display:grid;grid-template-columns:130px 1fr;gap:8px 20px;margin:16px 0}dt{font-weight:600}dd{margin:0;max-width:76ch}.stage{padding-top:2px;border-top:1px solid var(--line);margin-top:32px;scroll-margin-top:20px}.fixed{font-size:14px;color:var(--muted)}.decision{margin:14px 0}.route{display:flex;list-style:none;padding:0;margin:24px 0 28px;gap:24px}.route li{flex:1;min-width:0;position:relative;border-top:2px solid var(--line);padding-top:10px}.route li:not(:last-child)::after{content:'→';position:absolute;right:-20px;top:22px;color:var(--muted)}.route a{display:block;text-decoration:none;font-size:13px;color:var(--muted)}.route strong{display:block;color:var(--accent);font-size:16px;line-height:1.5;margin:6px 0}.route small{display:block;font-size:12px;line-height:1.6}.inline-links{display:flex;flex-wrap:wrap;gap:8px 24px;margin:12px 0}.caption{font-size:13px;color:var(--muted)}.config-line{font-weight:650;color:var(--accent);font-size:17px}
.acceptance-table td:first-child,.acceptance-table td:last-child{white-space:normal;text-align:left;min-width:180px;max-width:38ch}
@media(max-width:680px){main{padding:0 18px;margin-top:24px}h1{font-size:23px}h2{font-size:19px}.route{display:block;margin:18px 0}.route li{padding:12px 0 12px 20px;border-top:0;border-left:1px solid var(--line)}.route li:not(:last-child)::after{content:'↓';left:-6px;right:auto;top:auto;bottom:-13px;background:#fff}.route a{font-size:14px}.route strong{display:block;margin:4px 0}.route small{display:block}.route li:last-child{padding-bottom:0}dl{grid-template-columns:1fr;gap:2px}dd{margin-bottom:10px}th,td{padding:10px;font-size:13px}}
@media print{main{max-width:none;padding:0}a{color:inherit}.table-wrap{overflow:visible}.route{break-inside:avoid}details{break-inside:avoid}}
'''


def escape(value):
    return html.escape(str(value))


def page_start(title):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<link rel="icon" href="data:,"><title>' + escape(title) + '</title><style>' + STYLE
            + '</style></head><body><main><h1>' + escape(title) + '</h1>')


PAGE_END = '</main></body></html>'
