#!/usr/bin/env python3
"""Live guard: protocol-looking JSON string data survives every HTTP surface.

Usage: python3 tests/test_json_schema_protocol_routing.py http://127.0.0.1:8137
Requires a running server with a default model. No third-party dependencies.
"""
import json
import sys
import urllib.request

MARKERS = '<think>kept</think> </think:opensource> <|channel>thought <|channel|>final <|content_thinking|> to=self<|message|> <tool_call>literal</tool_call>'
SCHEMA = {
    'type': 'object',
    'properties': {'note': {'type': 'string', 'enum': [MARKERS]}},
    'required': ['note'],
    'additionalProperties': False,
}


def request(base, surface, streaming, thinking):
    cap = 2048 if thinking else 256
    body = {'model': 'mlx-serve', 'temperature': 0, 'stream': streaming}
    prompt = 'Return the JSON object required by the schema.'
    if surface == 'responses':
        body.update(max_output_tokens=cap, input=prompt, reasoning={'effort': 'medium' if thinking else 'none'},
                    text={'format': {'type': 'json_schema', 'name': 'markers', 'strict': True, 'schema': SCHEMA}})
    else:
        body.update(max_tokens=cap, messages=[{'role': 'user', 'content': prompt}])
        if surface == 'messages':
            body.update(thinking={'type': 'enabled' if thinking else 'disabled', 'budget_tokens': -1}, output_config={'format': {'type': 'json_schema', 'schema': SCHEMA}})
        else:
            body.update(enable_thinking=thinking, reasoning_budget_tokens=-1, response_format={'type': 'json_schema', 'json_schema': {'name': 'markers', 'strict': True, 'schema': SCHEMA}})
    req = urllib.request.Request(base + '/v1/' + surface, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read().decode()
    if not streaming:
        result = json.loads(raw)
        if surface == 'responses':
            content = ''.join(c['text'] for item in result['output'] if item['type'] == 'message' for c in item['content'] if c['type'] == 'output_text')
            reasoning = [item for item in result['output'] if item['type'] == 'reasoning' and item.get('summary')]
            tokens = result['usage']['output_tokens']
        elif surface == 'messages':
            content = ''.join(c['text'] for c in result['content'] if c['type'] == 'text')
            reasoning = [c for c in result['content'] if c['type'] == 'thinking' and c.get('thinking')]
            tokens = result['usage']['output_tokens']
        else:
            content = result['choices'][0]['message']['content']
            reasoning = result['choices'][0]['message'].get('reasoning_content')
            tokens = result['usage']['completion_tokens']
        assert 0 < tokens <= cap, result
    else:
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        assert events, raw
        if surface == 'responses':
            content = ''.join(e['delta'] for e in events if e.get('type') == 'response.output_text.delta')
            reasoning = [e for e in events if e.get('type') == 'response.reasoning_summary_text.delta' and e.get('delta')]
            assert any(e.get('type') == 'response.completed' for e in events), events
        elif surface == 'messages':
            content = ''.join(e['delta']['text'] for e in events if e.get('type') == 'content_block_delta' and e['delta']['type'] == 'text_delta')
            reasoning = [e for e in events if e.get('type') == 'content_block_delta' and e['delta']['type'] == 'thinking_delta' and e['delta'].get('thinking')]
            assert any(e.get('type') == 'message_stop' for e in events), events
        else:
            content = ''.join(c.get('delta', {}).get('content', '') or '' for e in events for c in e.get('choices', []))
            reasoning = [c['delta'].get('reasoning_content') for e in events for c in e.get('choices', []) if c.get('delta', {}).get('reasoning_content')]
            assert any(c.get('finish_reason') == 'stop' for e in events for c in e.get('choices', [])), events
    assert json.loads(content) == {'note': MARKERS}, content
    if not thinking:
        assert not reasoning, reasoning
    print(f'PASS: {surface}, stream={streaming}: exact marker data, thinking={thinking}', flush=True)


if __name__ == '__main__':
    for thinking in (False, True):
        for surface in ('chat/completions', 'messages', 'responses'):
            for streaming in (False, True):
                request(sys.argv[1].rstrip('/'), surface, streaming, thinking)
