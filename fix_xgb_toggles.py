f = r'c:\Users\deepjyotiray\source\repos\CTF\forextradernative\dashboard.html'
with open(f, 'r', encoding='utf-8') as fh:
    c = fh.read()

anchor = 'async function applySelectedProfileToLive(){'
new_fn = (
    'async function saveXgbToggle(key, enabled){\n'
    '  try{const r=await fetch(A+\'/config\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify({[key]:enabled})});const d=await r.json();if(d.config_version)_configVersion=d.config_version;setTimeout(fetchConfig,200);}catch(e){alert(\'Failed to save \'+key);}\n'
    '}\n'
)

if anchor in c:
    c = c.replace(anchor, new_fn + anchor)
    print('inserted saveXgbToggle')
else:
    print('anchor not found')

with open(f, 'w', encoding='utf-8') as fh:
    fh.write(c)
print('done')
