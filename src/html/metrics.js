'use strict';
// Live metrics panel — the markup AND the polling logic in one file. The panel
// is injected into the `#mlx-metrics` mount on the index page (only present when
// the server ran with --metrics), then this polls the open /metrics.json feed
// once a second. Everything here is NON-PERSISTED: the sparkline history lives
// only in these JS ring buffers, derived live from the counters/histograms the
// server already exposes (no server-side time series is stored).
//
// Decode & prefill "tok/s" are ACTIVE speeds — Δtokens ÷ Δ(phase time) over a
// trailing window. The phase-time sums (decode_time_seconds / prefill_time_
// seconds) and token counters only advance when a request COMPLETES, so this is
// the true generation/prefill speed of recently-finished requests — NOT the
// idle-averaged counter rate (which reads ~0 between requests).
// ── Pure rate math (no DOM, no module state) ─────────────────────────────────
//
// Every displayed rate is derived FROM THE CURRENT DATA on every tick. Nothing
// is carried between ticks. A value stashed in a module-level `let` outlives the
// condition that produced it: the Prefill tile used to keep showing the last
// prefill speed for the whole of a long decode, and only a page refresh cleared
// it — which is exactly the signature of state that isn't derived from the feed.
// Exported for `tests/metrics_panel_test.mjs`.

// Newest sample that is at least winMs old (tightest window >= winMs); the
// oldest retained sample while still warming up.
function panelAt(now, samples, winMs) {
  let s = samples[0];
  for (const x of samples) { if (now - x.t >= winMs) s = x; else break; }
  return s;
}

function computeRates(now, samples, c, g, psum) {
  const liveTok = (g.generation_tokens_live != null) ? g.generation_tokens_live : c.generation_tokens_total;
  const livePre = (g.prefill_tokens_live != null) ? g.prefill_tokens_live : 0;
  const prefilling = (g.requests_prefilling || 0) > 0;

  // Decode tok/s — LIVE decode speed while a request runs, 0 when idle. Tokens
  // only accrue during decode, so this is flat through prefill.
  let decodeTps = 0;
  if (g.requests_running > 0) {
    const wl = panelAt(now, samples, 4000);
    if (wl) {
      const dt = (now - wl.t) / 1000;
      if (dt > 0) decodeTps = Math.max(0, (liveTok - wl.live) / dt);
    }
  }

  // Prefill tok/s — LIVE prefill speed, 0 when no prefill is running. Same
  // no-carry-forward rule as decode: the big number answers "what is happening
  // NOW". Progress is published once per prefill CHUNK (8192 tokens), so the
  // window is wide enough to span one chunk even on a slow model.
  let prefillTps = 0;
  if (livePre > 0) {
    const wl = panelAt(now, samples, 30000);
    if (wl) {
      const dt = (now - wl.t) / 1000;
      if (dt > 0) prefillTps = Math.max(0, (livePre - wl.pre) / dt);
    }
  }

  // "How fast does this machine prefill?" — the stable answer, shown in the
  // sub-line where it can't be mistaken for a live rate. Cumulative, so it never
  // goes stale. Numerator is FORWARDED tokens (`prefill_tokens_total`), never
  // `prompt_tokens_total`: with the prefix cache warm most billed tokens are
  // restored, not computed, and dividing them by prefill time overstates
  // throughput by prompt/(prompt-cached) — measured 10.6x on a 35B MoE.
  const avgPrefillTps = (psum > 1e-6 && c.prefill_tokens_total > 0)
    ? c.prefill_tokens_total / psum
    : null;

  // Requests per second over a ~60s window.
  let reqRate = null;
  const wp = panelAt(now, samples, 60000);
  if (wp) {
    const dt = (now - wp.t) / 1000;
    if (dt > 0) reqRate = Math.max(0, (c.requests_success_total - wp.req) / dt);
  }

  return { decodeTps, prefillTps, avgPrefillTps, reqRate, prefilling, liveTok, livePre };
}

function fmt(v, d) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(d === undefined ? 1 : d);
}

// The Prefill sub-line while a prefill is in flight. Pure, so every state below
// is testable without a DOM.
//
// `prefill_tokens_live` is a numerator with no denominator on its own: "6.0K
// tokens done" out of WHAT? `prefill_target_tokens` is that denominator, and it
// is on the SAME base — the post-prefix-cache-trim length the chunk loop will
// really forward — so live/target is a true percentage rather than a fraction of
// a prompt most of which was never computed. `prompt - target` is what the hot
// cache restored for THIS request, which the lifetime `prefix_cache_*_total`
// counters cannot answer.
//
// target == 0 is NOT 0%. It means either the cache restore is still running (the
// target is published after the trim, and the restore takes seconds on a long
// prompt) or this is a ds4/llama/diffusion prefill, which never moves any of the
// three gauges. Rendering "0%" there would claim the work has a known size and
// has not started; both are false. Fall back to the bare phase word instead.
function prefillPhaseLabel(g) {
  const live = (g.prefill_tokens_live != null) ? g.prefill_tokens_live : 0;
  const target = g.prefill_target_tokens || 0;
  const prompt = g.prefill_prompt_tokens || 0;
  if (target <= 0) return 'prefilling' + (live > 0 ? ' · ' + fmt(live, 0) + ' tok' : '');
  // Clamped: the chunk loop stops a token (plus the SSM snapshot backoff) short
  // of the target, and a sampler tick can pair a new request's live count with
  // the previous target. Neither should ever render as 101%.
  const pct = Math.min(100, Math.max(0, Math.round((live / target) * 100)));
  const reused = prompt > target ? prompt - target : 0;
  return 'prefilling · ' + fmt(live, 0) + ' / ' + fmt(target, 0) + ' (' + pct + '%)' +
         (reused > 0 ? ' · ' + fmt(reused, 0) + ' reused' : '');
}

// Node (tests) sees no `document`; the browser sees no `globalThis.__mlxPanel`
// consumer. Either way the IIFE below only runs in a real page.
if (typeof globalThis !== 'undefined') globalThis.__mlxPanel = { computeRates, panelAt, prefillPhaseLabel, fmt };

if (typeof document !== 'undefined') (function () {
  // Panel markup, injected into the page. A template literal, so the CSS/HTML
  // braces need no escaping — the reason this lives here and not inline in the
  // std.fmt-formatted index.html.
  const PANEL_HTML = `
<style>
.mhead{display:flex;align-items:center;gap:10px;margin:24px 0 10px}
.mhead h2{margin:0}
#m-status{font-size:11px;font-weight:600;letter-spacing:.02em;padding:2px 9px;border-radius:999px;background:#1a1e25;color:#7d8794}
#m-status.live{background:#0f2a17;color:#4ade80}
#m-status.err{background:#2a0f14;color:#ff95a8}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:640px){.mgrid{grid-template-columns:repeat(2,1fr)}}
.mtile{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:12px 14px}
.mlbl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#7d8794;margin-bottom:7px}
.mval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:26px;font-weight:700;line-height:1;color:#e6e9ee}
.munit{font-size:12px;font-weight:400;color:#7d8794;margin-left:4px}
.msub{font-size:11px;color:#5b6470;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mbar{height:5px;background:#1f242c;border-radius:3px;margin-top:10px;overflow:hidden}
.mfill{height:100%;width:0;border-radius:3px;background:#3b82f6;transition:width .5s}
.mfill.warn{background:#f59e0b}.mfill.crit{background:#ef4444}
.mspark{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:640px){.mspark{grid-template-columns:1fr}}
.msparkbox{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:10px 12px}
.msparkhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.msparkval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;font-weight:700;color:#e6e9ee}
.msparkbox svg{width:100%;height:44px;display:block}
</style>
<div class=mhead><h2 style="margin:0">Live metrics</h2><span id=m-status>connecting…</span></div>
<div class=card style="padding:16px">
<div class=mgrid>
<div class=mtile><div class=mlbl>Decode</div><div class=mval><span id=m-decode-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-decode-ms>— ms avg</div></div>
<div class=mtile><div class=mlbl>Prefill</div><div class=mval><span id=m-prefill-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-prefill-ms>— ms avg</div></div>
<div class=mtile><div class=mlbl>Requests</div><div class=mval><span id=m-running>0</span><span class=munit>running</span></div><div class=msub id=m-waiting>0 waiting · — req/s</div></div>
<div class=mtile><div class=mlbl>Avg TTFT</div><div class=mval><span id=m-ttft>—</span><span class=munit>ms</span></div><div class=msub id=m-e2e>— ms e2e</div></div>
<div class=mtile><div class=mlbl>Cache hit rate</div><div class=mval><span id=m-cache>—</span><span class=munit>%</span></div><div class=msub id=m-cachedetail>— / — queries</div></div>
<div class=mtile><div class=mlbl>GPU</div><div class=mval><span id=m-gpu>0</span><span class=munit>%</span></div><div class=mbar><div class=mfill id=m-gpubar></div></div></div>
<div class=mtile><div class=mlbl>Memory</div><div class=mval><span id=m-mem>0</span><span class=munit>MB</span></div><div class=msub>physical footprint</div></div>
<div class=mtile><div class=mlbl>Generated</div><div class=mval><span id=m-gen>0</span><span class=munit>tok</span></div><div class=msub id=m-success>0 requests</div></div>
</div>
<div class=mspark>
<div class=msparkbox><div class=msparkhead><span class=mlbl>Decode tok/s · last 60s</span><span class=msparkval id=m-spark-decode-val>—</span></div><svg id=m-spark-decode viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl>Prefill tok/s · last 60s</span><span class=msparkval id=m-spark-prefill-val>—</span></div><svg id=m-spark-prefill viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
</div>
</div>`;

  const mount = document.getElementById('mlx-metrics');
  if (mount) mount.innerHTML = PANEL_HTML;

  const $ = (id) => document.getElementById(id);
  const samples = [];              // {t, gen, dsum, prompt, psum, req} ring buffer
  const RETAIN_MS = 120000;        // keep 2 min of samples for the 60s window
  const SPARK_N = 60;              // sparkline points (≈60s at 1 Hz)
  const decodeHist = [], prefillHist = [];
  const hover = { decode: null, prefill: null };  // hovered point index per chart

  function setStatus(cls, txt) {
    const e = $('m-status');
    if (e) { e.className = cls; e.textContent = txt; }
  }


  // Draw a sparkline (auto-scaled) into an <svg>, set its value label, and — when
  // the mouse is over that chart (hover[key] set) — draw a marker at the hovered
  // point and show that point's value in the label instead of the latest.
  function spark(id, data, color, valId, key, dec) {
    const svg = $(id), label = $(valId);
    if (!svg) return;
    if (data.length < 2) {
      svg.innerHTML = '';
      if (label) label.textContent = data.length ? fmt(data[0], dec) : '—';
      return;
    }
    const max = Math.max.apply(null, data.concat([0.001]));
    const n = data.length, W = 300, H = 44, p = 3;
    const xs = new Array(n), ys = new Array(n);
    let pts = '';
    for (let i = 0; i < n; i++) {
      xs[i] = p + (i / (n - 1)) * (W - 2 * p);
      ys[i] = H - p - (data[i] / max) * (H - 2 * p);
      pts += (i ? ' ' : '') + xs[i].toFixed(1) + ',' + ys[i].toFixed(1);
    }
    let m =
      '<polyline points="' + pts + '" fill="none" stroke="' + color +
      '" stroke-width="1.5" stroke-linejoin="round"/>' +
      '<line x1="' + p + '" y1="' + (H - 1) + '" x2="' + (W - p) + '" y2="' + (H - 1) +
      '" stroke="#1f242c" stroke-width="1"/>';
    const hi = hover[key];
    if (hi !== null && hi >= 0 && hi < n) {
      m += '<line x1="' + xs[hi].toFixed(1) + '" y1="' + p + '" x2="' + xs[hi].toFixed(1) +
        '" y2="' + (H - 1) + '" stroke="' + color + '" stroke-width="1" opacity="0.35"/>' +
        '<circle cx="' + xs[hi].toFixed(1) + '" cy="' + ys[hi].toFixed(1) + '" r="2.5" fill="' + color + '"/>';
      if (label) label.textContent = fmt(data[hi], dec);
    } else if (label) {
      label.textContent = fmt(data[n - 1], dec);
    }
    svg.innerHTML = m;
  }

  // Wire mouse hover on a sparkline: map cursor x → nearest data point, mark it,
  // show that value in the chart's label; mouseleave restores the latest value.
  function attachSparkHover(id, key, getData, color, valId, dec) {
    const svg = $(id);
    if (!svg) return;
    svg.style.cursor = 'crosshair';
    svg.addEventListener('mousemove', function (e) {
      const data = getData();
      if (data.length < 2) return;
      const rect = svg.getBoundingClientRect();
      const frac = rect.width ? (e.clientX - rect.left) / rect.width : 0;
      hover[key] = Math.max(0, Math.min(data.length - 1, Math.round(frac * (data.length - 1))));
      spark(id, data, color, valId, key, dec);
    });
    svg.addEventListener('mouseleave', function () {
      hover[key] = null;
      spark(id, getData(), color, valId, key, dec);
    });
  }

  const histSum = (hist) => (hist && typeof hist.sum === 'number') ? hist.sum : 0;

  async function tick() {
    let d;
    try {
      const r = await fetch('/metrics.json', { cache: 'no-store' });
      if (r.status === 503) { setStatus('err', 'metrics disabled'); return; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      d = await r.json();
    } catch (e) { setStatus('err', 'error: ' + e.message); return; }

    setStatus('live', '● live');
    const c = d.counters, g = d.gauges, h = d.histograms;
    const now = Date.now();

    const psum = histSum(h.prefill_time_seconds);
    const liveTok = (g.generation_tokens_live != null) ? g.generation_tokens_live : c.generation_tokens_total;
    const livePre = (g.prefill_tokens_live != null) ? g.prefill_tokens_live : 0;
    samples.push({ t: now, live: liveTok, pre: livePre, pretok: c.prefill_tokens_total, psum: psum, req: c.requests_success_total });
    while (samples.length > 2 && now - samples[0].t > RETAIN_MS) samples.shift();

    // Everything displayed is derived here, from THIS tick's data. Nothing is
    // remembered between ticks — see the note above `computeRates`.
    const r = computeRates(now, samples, c, g, psum);
    const { decodeTps, prefillTps, avgPrefillTps, reqRate, prefilling } = r;

    // Sparkline history: both series dip to 0 when their phase is idle.
    decodeHist.push(decodeTps); if (decodeHist.length > SPARK_N) decodeHist.shift();
    prefillHist.push(prefillTps); if (prefillHist.length > SPARK_N) prefillHist.shift();

    // Average latency from each histogram's sum/count (seconds → ms).
    const avgMs = (hist) => (hist && hist.count > 0) ? (hist.sum / hist.count) * 1000 : null;
    const ttft = avgMs(h.time_to_first_token_seconds);
    const e2e = avgMs(h.e2e_request_latency_seconds);
    const decodeMs = avgMs(h.decode_time_seconds);
    const prefillMs = avgMs(h.prefill_time_seconds);

    const cq = c.prefix_cache_queries_total, ch = c.prefix_cache_hits_total;
    const cachePct = cq > 0 ? Math.round((ch / cq) * 100) : null;
    // Token-level reuse: what fraction of billed prompt tokens never reached the
    // GPU. This is the number that explains a low prefill tok/s on warm turns.
    const tokTotal = c.prompt_tokens_total;
    const tokPct = tokTotal > 0 ? Math.round((c.prefix_cache_tokens_total / tokTotal) * 100) : null;

    $('m-decode-tps').textContent = fmt(decodeTps, 1);
    $('m-decode-ms').textContent = (decodeMs !== null ? fmt(decodeMs, 0) : '—') + ' ms avg';
    // Big number = live prefill speed, 0 when not prefilling (mirrors Decode).
    $('m-prefill-tps').textContent = fmt(prefillTps, 0);
    // Sub-line doubles as the phase indicator AND carries the stable average, so
    // "0 tok/s while decoding" never means "I don't know how fast prefill is".
    // The phase flag flips at prefill START; the token count appears once the
    // first chunk lands (and never for ds4/llama, which prefill elsewhere).
    $('m-prefill-ms').textContent = prefilling
      ? prefillPhaseLabel(g)
      : (avgPrefillTps !== null
          ? (fmt(avgPrefillTps, 0) + ' tok/s avg · ' + (prefillMs !== null ? fmt(prefillMs, 0) : '—') + ' ms')
          : '— ms avg');
    $('m-running').textContent = g.requests_running;
    $('m-waiting').textContent = g.requests_waiting + ' waiting · ' + (reqRate !== null ? fmt(reqRate, 2) : '—') + ' req/s';
    $('m-ttft').textContent = ttft !== null ? fmt(ttft, 0) : '—';
    $('m-e2e').textContent = (e2e !== null ? fmt(e2e, 0) : '—') + ' ms e2e';
    $('m-cache').textContent = cachePct !== null ? cachePct : '—';
    $('m-cachedetail').textContent = ch + ' / ' + cq + ' queries'
      + (tokPct !== null ? ' · ' + tokPct + '% tokens reused' : '');

    const gp = g.gpu_utilization_pct;
    $('m-gpu').textContent = gp;
    const bar = $('m-gpubar');
    bar.style.width = gp + '%';
    bar.className = 'mfill' + (gp >= 90 ? ' crit' : gp >= 70 ? ' warn' : '');

    $('m-mem').textContent = g.memory_mb;
    // Live count (completed + in-flight) so it moves during a running request.
    $('m-gen').textContent = fmt(liveTok, 0);
    $('m-success').textContent = fmt(c.requests_success_total, 0) + ' requests';

    spark('m-spark-decode', decodeHist, '#22c55e', 'm-spark-decode-val', 'decode', 1);
    spark('m-spark-prefill', prefillHist, '#3b82f6', 'm-spark-prefill-val', 'prefill', 0);
  }

  attachSparkHover('m-spark-decode', 'decode', function () { return decodeHist; }, '#22c55e', 'm-spark-decode-val', 1);
  attachSparkHover('m-spark-prefill', 'prefill', function () { return prefillHist; }, '#3b82f6', 'm-spark-prefill-val', 0);
  tick();
  setInterval(tick, 1000);
})();
