#!/usr/bin/env python3
"""Execute tool calls and replay their results without returned reasoning.

Run against an isolated ds4 server; BENCH_API_KEY authenticates requests.
The generated code is restricted to arithmetic and print by AST validation.
"""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import urllib.request


def execute(code):
    tree = ast.parse(code)
    allowed = (ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
               ast.Call, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        assert isinstance(node, allowed), f'Unexpected code: {ast.dump(node)}'
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == 'print'
    output = io.StringIO()
    scope = {'__builtins__': {}, 'print': print}
    value = None
    with contextlib.redirect_stdout(output):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = tree.body.pop()
            exec(compile(tree, '<tool>', 'exec'), scope)
            value = eval(compile(ast.Expression(tail.value), '<tool>', 'eval'), scope)
        else:
            exec(compile(tree, '<tool>', 'exec'), scope)
    return output.getvalue().strip() or (str(value) if value is not None else '')


def call(base, payload):
    key = os.environ.get('BENCH_API_KEY', '')
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
    request = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        if not payload['stream']:
            return json.load(response)
        message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
        calls, usage = {}, {}
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if not data or data == b'[DONE]':
                continue
            event = json.loads(data)
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                for field in ('content', 'reasoning_content'):
                    message[field] += delta.get(field) or ''
                for part in delta.get('tool_calls', []):
                    item = calls.setdefault(part['index'], {'id': '', 'type': 'function',
                                                           'function': {'name': '', 'arguments': ''}})
                    item['id'] = part.get('id') or item['id']
                    for field in ('name', 'arguments'):
                        item['function'][field] += part.get('function', {}).get(field) or ''
        if calls:
            message['tool_calls'] = [calls[i] for i in sorted(calls)]
        return {'choices': [{'message': message}], 'usage': usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tools = [{'type': 'function', 'function': {'name': 'execute_code',
              'description': 'Run Python arithmetic; return stdout or the final expression value.',
              'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                             'required': ['code']}}}]
    records = []
    for i, (a, b) in enumerate(((17, 23), (19, 37), (41, 13), (29, 31), (43, 47))):
        messages = [{'role': 'system', 'content': 'Immediately use execute_code to calculate. '
                     'After its result, reply only VERIFIED followed by that number.'},
                    {'role': 'user', 'content': f'Calculate {a} * {b} using the tool.'}]
        payload = {'model': args.model, 'messages': messages, 'tools': tools,
                   'max_tokens': 512, 'temperature': 0, 'enable_thinking': True,
                   'stream': bool(i % 2), 'stream_options': {'include_usage': True}}
        first = call(args.base_url, payload)
        record = {'request': payload.copy(), 'first': first}
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2))
        assistant = first['choices'][0]['message']
        calls = assistant.get('tool_calls', [])
        assert len(calls) == 1, assistant
        tool = calls[0]
        assert tool['function']['name'] == 'execute_code', tool
        code = json.loads(tool['function']['arguments'])['code']
        assert isinstance(code, str) and code.strip(), tool
        output = execute(code)
        assert output == str(a * b), output
        replay = {k: assistant[k] for k in ('role', 'content', 'tool_calls') if k in assistant}
        payload['messages'] = messages + [replay, {'role': 'tool',
                               'tool_call_id': tool['id'], 'content': output}]
        second = call(args.base_url, payload)
        record.update(second=second, executed_output=output)
        args.output.write_text(json.dumps(records, indent=2))
        final = second['choices'][0]['message']
        assert not final.get('tool_calls'), final
        assert ' '.join((final.get('content') or '').split()) == 'VERIFIED ' + output, final
        cached = second.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
        assert cached > first.get('usage', {}).get('prompt_tokens', 0), second.get('usage')
        print(f'PASS case={i + 1} stream={payload["stream"]} result={output} cached={cached}', flush=True)


if __name__ == '__main__':
    main()
