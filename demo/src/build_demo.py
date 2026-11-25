# usage: python3 -I build_demo.py <demo_dir> <offline_html_out> <artifact_html_out>
import sys, json
D, OFF, ART = sys.argv[1:4]
app = open(f'{D}/app.html').read()
model = open(f'{D}/model.json').read().replace('</', '<\\/')
data = open(f'{D}/demo_data.json').read().replace('</', '<\\/')
core = open(f'{D}/wfm_core.js').read()
for k, v in [('%%MODEL%%', model), ('%%DATA%%', data), ('%%CORE%%', core)]:
    assert app.count(k) == 1; app = app.replace(k, v)
open(ART, 'w').write(app)
cut = app.index('</style>') + len('</style>')
head, body = app[:cut], app[cut:]
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')  # artifact-viewer tag, not needed offline
doc = ('<!doctype html>\n<html lang="en" data-theme="light">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + '\n</head>\n<body>' + body + '\n</body>\n</html>\n')
open(OFF, 'w').write(doc)
print('offline', round(len(doc) / 1e3), 'KB; artifact', round(len(app) / 1e3), 'KB')
