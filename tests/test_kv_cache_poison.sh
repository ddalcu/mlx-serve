#!/bin/bash
# test_kv_cache_poison.sh — a pad-only generation must not poison the KV cache: request A answers, request B (a long
# noisy tool result) produces only <pad>, then A again must still answer as on a fresh server.
#
#   ./tests/test_kv_cache_poison.sh [port]
#
# A failure means pad-only state was committed to the prefix cache (commitDeclinesPadOnly).

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"

echo "=== KV Cache Poisoning Test ==="
echo "Server: $BASE"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

# Helper: send a request, return completion_tokens
send_request() {
    local label="$1"
    local body="$2"
    local result
    result=$(curl -sf "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null)
    local tokens
    tokens=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
    local content
    content=$(echo "$result" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message'].get('content',''); print(c.replace('<pad>','').strip()[:80])" 2>/dev/null)
    echo "$tokens|$content"
}

# Request A: Simple tool response that should always work
REQUEST_A='{
    "model": "mlx-serve",
    "messages": [
        {"role": "system", "content": "You are helpful. Be brief."},
        {"role": "user", "content": "What day is today?"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"date\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Fri Apr 4 17:00:00 PDT 2026"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
    "max_tokens": 256, "temperature": 0.7, "stream": false
}'

# Request B: Long noisy tool response that may cause pad-only generation on small models
REQUEST_B='{
    "model": "mlx-serve",
    "messages": [
        {"role": "system", "content": "You are a helpful macOS assistant. Use tools for tasks. Answer directly when no tools needed. For web search use webSearch tool."},
        {"role": "user", "content": "search the web for finance news and give me the headlines"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_99", "type": "function", "function": {"name": "webSearch", "arguments": "{\"query\":\"finance news headlines\"}"}}]},
        {"role": "tool", "tool_call_id": "call_99", "content": "Navigated to https://html.duckduckgo.com/html/?q=finance%20news%20headlines\nTitle: finance news headlines at DuckDuckGo\n\nPage content:\nStock Markets, Business News, Financials, Earnings - CNBC\n \n\n \n\n \n \n \n \n \n \n \n \n \n www.cnbc.com\n \n \n     2026-04-04T00:00:00.0000000\n \n \n \n \n\n \n \n CNBC is the world leader in business news and real-time financial market coverage. Find fast, actionable information.\n \n \n\n \n \n \n \n \n \n \n \n \n \n Latest Finance News | Today s Top Headlines | Reuters\n \n\n \n\n \n \n \n \n \n \n \n \n \n www.reuters.com/business/finance/\n \n \n     2026-04-02T00:00:00.0000000\n \n \n \n \n\n \n \n Find latest finance news from every corner of the globe at Reuters.com, your online source for breaking international news coverage.\n \n \n\n \n \n \n \n \n \n \n \n \n \n MarketWatch: Stock Market News - Financial News\n \n\n \n\n \n \n \n \n \n \n \n \n \n www.marketwatch.com\n \n \n \n \n \n\n \n \n MarketWatch provides the latest stock market, financial and business news. Get stock market quotes, personal finance advice, company news and more."}
    ],
    "tools": [
        {"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
        {"type":"function","function":{"name":"webSearch","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
        {"type":"function","function":{"name":"readFile","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
        {"type":"function","function":{"name":"writeFile","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
        {"type":"function","function":{"name":"editFile","description":"Find and replace in file","parameters":{"type":"object","properties":{"path":{"type":"string"},"find":{"type":"string"},"replace":{"type":"string"}},"required":["path","find","replace"]}}},
        {"type":"function","function":{"name":"searchFiles","description":"Grep for pattern","parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}},
        {"type":"function","function":{"name":"browse","description":"Browse a URL","parameters":{"type":"object","properties":{"action":{"type":"string"},"url":{"type":"string"}},"required":["action"]}}}
    ],
    "max_tokens": 8192, "temperature": 0.7, "top_p": 0.95, "stream": false
}'

# Step 1: Request A on fresh server (should work)
echo ""
echo "--- Step 1: Request A (fresh server) ---"
IFS='|' read -r TOKENS1 CONTENT1 <<< "$(send_request "A_fresh" "$REQUEST_A")"
echo "  tokens: $TOKENS1"
echo "  content: '$CONTENT1'"
if [ -z "$CONTENT1" ] || [ "$TOKENS1" -le 2 ] 2>/dev/null; then
    echo "  FAIL: Request A doesn't work even on fresh server. Cannot test cache poisoning."
    echo "  (This model may not support tool calling at all)"
    exit 0
fi
echo "  PASS: Request A works on fresh server"

# Step 2: Request B (may produce pad-only output)
echo ""
echo "--- Step 2: Request B (long noisy tool result) ---"
IFS='|' read -r TOKENS2 CONTENT2 <<< "$(send_request "B_noisy" "$REQUEST_B")"
echo "  tokens: $TOKENS2"
echo "  content: '$CONTENT2'"
if [ -n "$CONTENT2" ] && [ "$TOKENS2" -gt 2 ] 2>/dev/null; then
    echo "  Request B succeeded — model handles this content. No cache poisoning to test."
    exit 0
fi
echo "  Request B produced pad-only output (expected for some models)"

# Step 3: Request A again (THE TEST — should still work if cache isn't poisoned)
echo ""
echo "--- Step 3: Request A again (after pad failure — THE TEST) ---"
IFS='|' read -r TOKENS3 CONTENT3 <<< "$(send_request "A_after_poison" "$REQUEST_A")"
echo "  tokens: $TOKENS3"
echo "  content: '$CONTENT3'"
if [ -n "$CONTENT3" ] && [ "$TOKENS3" -gt 2 ] 2>/dev/null; then
    echo "  PASS: Request A still works after pad failure — cache not poisoned"
else
    echo "  FAIL: Request A produces pad-only output after prior pad failure"
    echo ""
    echo "  ROOT CAUSE: KV cache poisoned by prior pad-only generation."
    echo "  The server stores the KV cache state after a failed generation,"
    echo "  and subsequent requests reuse the corrupted cache."
    echo "  Fix: Skip updateCachedPrompt when completion_tokens <= pad threshold."
    exit 1
fi

echo ""
echo "=== All tests passed ==="
