#!/bin/bash
# Vision recognition test for mlx-serve
# Requires: running mlx-serve with a Gemma 4 vision model, Python 3 with PIL
# Usage: ./tests/test_vision.sh [port]
set -euo pipefail

PORT="${1:-8080}"
API="http://127.0.0.1:$PORT/v1/chat/completions"
FIXTURES="$(dirname "$0")/fixtures"
PASS=0; FAIL=0; TOTAL=0

# Check server is running
if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server not running on port $PORT"
    exit 1
fi

# Check fixture images exist
for f in house.jpeg street-name-signs.jpg robot.png not_hot_dog_app.webp; do
    if [ ! -f "$FIXTURES/$f" ]; then
        echo "ERROR: Missing fixture image: $FIXTURES/$f"
        exit 1
    fi
done

run_test() {
    TOTAL=$((TOTAL+1))
    local name="$1"; local result="$2"; local pattern="$3"
    if echo "$result" | grep -qiE "$pattern"; then
        PASS=$((PASS+1)); echo "  ✓ $name → $(echo "$result" | head -1 | cut -c1-60)"
    else
        FAIL=$((FAIL+1)); echo "  ✗ $name → $(echo "$result" | head -1 | cut -c1-60)"
        echo "    expected: $pattern"
    fi
}

send_image() {
    local filepath="$1"; local prompt="$2"; local max_tokens="${3:-30}"
    python3 -c "
import base64, json, urllib.request
with open('$filepath', 'rb') as f: img = f.read()
ext = '$filepath'.rsplit('.', 1)[-1].lower()
mime = {'jpeg':'image/jpeg','jpg':'image/jpeg','png':'image/png','webp':'image/webp'}.get(ext, 'image/jpeg')
b64 = base64.b64encode(img).decode()
msg = {'model':'test','max_tokens':$max_tokens,'temperature':0.0,'stream':False,
       'messages':[{'role':'user','content':[
           {'type':'image_url','image_url':{'url':f'data:{mime};base64,{b64}'}},
           {'type':'text','text':'''$prompt'''}]}]}
req = urllib.request.Request('$API', json.dumps(msg).encode(), {'Content-Type':'application/json'})
resp = urllib.request.urlopen(req, timeout=180)
print(json.loads(resp.read())['choices'][0]['message']['content'].strip())
" 2>/dev/null
}

echo "═══════════════════════════════════════════════════════════"
echo " Vision Recognition Test — mlx-serve (port $PORT)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "── house.jpeg ──"
R=$(send_image "$FIXTURES/house.jpeg" "What color is this house? One word only.")
run_test "House color" "$R" "blue"

echo ""
echo "── street-name-signs.jpg ──"
R=$(send_image "$FIXTURES/street-name-signs.jpg" "What shape is the red sign? One word.")
run_test "Stop sign shape" "$R" "octagon"

R=$(send_image "$FIXTURES/street-name-signs.jpg" "What are the street names on the signs? List them." 50)
run_test "Street name reading" "$R" "grey.?fox|waterfall"

echo ""
echo "── robot.png ──"
R=$(send_image "$FIXTURES/robot.png" "Is this a photograph or a digital render? One word.")
run_test "Render detection" "$R" "render|digital|illustration|cgi|3d"

R=$(send_image "$FIXTURES/robot.png" "Does this show a human or a robot? One word.")
run_test "Robot detection" "$R" "robot"

echo ""
echo "── not_hot_dog_app.webp ──"
R=$(send_image "$FIXTURES/not_hot_dog_app.webp" "What food item do you see? One word." 30)
run_test "WebP food recognition" "$R" "hot.?dog|sausage|frank"

echo ""
echo "── Format support ──"
# Just verify each format decodes without error
for f in house.jpeg street-name-signs.jpg robot.png not_hot_dog_app.webp; do
    R=$(send_image "$FIXTURES/$f" "What is this?" 10)
    if [ -n "$R" ] && ! echo "$R" | grep -qi "error"; then
        TOTAL=$((TOTAL+1)); PASS=$((PASS+1))
        echo "  ✓ $f decoded OK"
    else
        TOTAL=$((TOTAL+1)); FAIL=$((FAIL+1))
        echo "  ✗ $f decode FAILED"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Results: $PASS/$TOTAL passed ($FAIL failed)"
echo "═══════════════════════════════════════════════════════════"
exit $FAIL
