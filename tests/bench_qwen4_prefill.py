#!/usr/bin/env python3
"""Fresh-process llmprobe arms on one frozen binary; saves launches and engagement."""
import argparse, hashlib, json, os, re, socket, subprocess, time, urllib.request
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--lock', type=Path, required=True)
p.add_argument('--rungs', default='4k,8k,16k,64k,128k')
p.add_argument('--control-rungs', default='64k', help='rungs for final OFF drift-control arms')
p.add_argument('--runs', type=int, default=3)
p.add_argument('--arms', default='off,on')
p.add_argument('--extra-on', action='append', default=[], help='additional KEY=VALUE on candidate arms')
p.add_argument('--port', type=int, default=11234)
p.add_argument('--ctx', type=int, default=262144)
a = p.parse_args()
binary = a.binary.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
root = binary.parents[2]
model = 'ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit'
base = f'http://127.0.0.1:{a.port}'
def listening():
    with socket.socket() as s:
        s.settimeout(.2)
        return s.connect_ex(('127.0.0.1', a.port)) == 0
def props():
    with urllib.request.urlopen(base + '/props') as r: return json.load(r)
def request(text):
    body = {'model':model, 'messages':[{'role':'user','content':text}], 'temperature':0,
            'max_tokens':16, 'seed':1234, 'enable_mtp':False, 'enable_pld':False}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)
if listening(): raise RuntimeError('Benchmark port is occupied')
meta = {'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(), 'model':model,
        'llmprobe':'0.6.6', 'power':subprocess.check_output(['pmset','-g','batt'], text=True), 'arms':[]}
def save(): (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
owner = 'qwen4-prefill-' + str(os.getpid())
subprocess.run([str(a.lock.resolve()), 'acquire', owner], check=True)
try:
    for i, arm in enumerate(a.arms.split(',')):
        if arm not in ('off','on'): raise ValueError(arm)
        value = '1' if arm == 'on' else '0'; label = f'{i}-{arm}'
        env = os.environ.copy()
        # No inherited experimental/profiling switches can accidentally select an arm.
        for key in list(env):
            if key.startswith(('MLX_SERVE_', 'QWEN4_')): env.pop(key)
        settings = dict(MLX_SERVE_PREFILL_CHUNK='8192', MLX_SERVE_PREFILL_TRACE='1',
                        MLX_SERVE_QSA_PAIR=value, MLX_SERVE_HC_PREFILL=value,
                        MLX_SERVE_GDN_PREFILL_FUSED=value, MLX_SERVE_PLE_PACKED='0',
                        MLX_SERVE_PLE_AHEAD='0', MLX_SERVE_MOE_PREFILL_GROUP='0')
        if arm == 'on':
            for item in a.extra_on:
                key, sep, val = item.partition('=')
                if not sep or not key.startswith(('MLX_SERVE_', 'QWEN4_')): raise ValueError(item)
                settings[key] = val
        env.update(settings)
        env['DYLD_LIBRARY_PATH'] = str(root/'lib/mlx/lib')+':'+str(root/'lib/llama/lib')
        cmd = [str(binary),'serve','--host','127.0.0.1','--port',str(a.port),
               '--ctx-size',str(a.ctx),'--prefill-chunk','8192','--prefix-cache-entries','0','--kv-quant','off','--no-mtp','--no-pld','--no-drafter']
        probe = ['npx','--offline','llmprobe@0.6.6',base,'-m',model,'--bench-only',
                 '--rungs',(a.control_rungs if i >= 2 and arm == 'off' else a.rungs),'--runs',str(a.runs),'--no-save','--no-color',
                 '--save',str(out/(label+'.json')),'--label',label]
        rec = {'label':label,'launch':cmd,'env':settings,'probe':probe}; meta['arms'].append(rec); save()
        logpath = out/(label+'-server.log')
        with logpath.open('w') as log:
            server = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            try:
                start = time.monotonic()
                while not listening():
                    if server.poll() is not None: raise RuntimeError(logpath.read_text()[-4000:])
                    if time.monotonic()-start > 30: raise TimeoutError('bind')
                    time.sleep(.2)
                ready = request('Capital of France? Answer with one word.')
                assert 'paris' in ready['choices'][0]['message']['content'].lower(), ready
                rec['memory_before'] = props().get('memory',{}); save()
                with (out/(label+'-llmprobe.log')).open('w') as plog:
                    subprocess.run(probe, stdout=plog, stderr=subprocess.STDOUT, check=True)
                rec['memory_after'] = props().get('memory',{})
                lines = logpath.read_text().splitlines()
                rec['engagement'] = [s for s in lines if any(x in s for x in ('[qsa-pair]', '[hc-prefill]', '[gdn-prefill]', '[moe-prefill-group]', '[ple-ahead]', '[prefill-cadence]'))]
                rec['timings'] = [s for s in lines if 'prefill:' in s]
                save()
                if arm == 'on':
                    for marker in ('[qsa-pair] engaged:', '[hc-prefill] engaged:', '[gdn-prefill] engaged:'):
                        assert any(marker in s for s in lines), marker
                print(json.dumps({'finished':label,'report':str(out/(label+'.json'))}), flush=True)
            finally:
                server.terminate()
                try: server.wait(timeout=20)
                except subprocess.TimeoutExpired: server.kill(); server.wait()
                limit=time.monotonic()+10
                while listening() and time.monotonic()<limit: time.sleep(.2)
                if listening(): raise RuntimeError('Owned port did not close')
finally:
    subprocess.run([str(a.lock.resolve()), 'release', owner], check=True)
