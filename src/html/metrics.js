'use strict';

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

function monitorWindow(data, now, windowMs, model) {
  const m = data.monitor || {}, start = windowMs === 'startup' ? m.server?.started_at_ms : now - windowMs;
  if (!Number.isFinite(start) || start > now) return { requests: [], history: [], percentile: () => null, rate: () => null, partial: false };
  const requests = (m.recent_requests || []).filter(r => r.finished_at_ms >= start && r.finished_at_ms <= now && (!model || r.model === model));
  const history = monitorHistory(m).filter(s => s.at_ms >= start && s.at_ms <= now);
  const values = key => requests.filter(r => r.outcome === 'success').map(r => r[key]).filter(v => typeof v === 'number' && Number.isFinite(v) && v >= 0).sort((a,b) => a-b);
  const percentile = (key, q) => { const v = values(key); return v.length ? v[Math.max(0, Math.ceil(v.length*q)-1)] : null; };
  const rate = key => { if (history.length < 2 || model) return null; const a=history[0], b=history[history.length-1], dt=(b.at_ms-a.at_ms)/1000; return dt > 0 && b[key] >= a[key] ? (b[key]-a[key])/dt : null; };
  const oldest = (m.recent_requests || [])[0];
  return { requests, history, percentile, rate, partial: !!(m.retention?.requests_dropped && oldest && oldest.finished_at_ms > start) };
}
function monitorHistory(m) {
  const raw = m.history || [], oldest = raw[0]?.at_ms;
  return (m.history_archive || []).filter(s => Number.isFinite(s.at_ms) && (oldest == null || s.at_ms < oldest)).concat(raw);
}
function monitorValidPair(a, b, sampleIntervalMs) {
  const dt = b.at_ms - a.at_ms;
  if (!Number.isFinite(dt) || dt <= 0) return false;
  if (Number.isSafeInteger(a.continuity_id) && Number.isSafeInteger(b.continuity_id)) return a.continuity_id === b.continuity_id;
  return dt <= sampleIntervalMs * 3;
}
function monitorWindowCounters(history, keys, start, end, sampleIntervalMs=2000) {
  const deltas=Object.fromEntries(keys.map(key=>[key,0]));
  let coverageMs=0;
  for(let i=1;i<history.length;i++) {
    const a=history[i-1], b=history[i], dt=b.at_ms-a.at_ms;
    if(a.at_ms<start||b.at_ms>end||!monitorValidPair(a,b,sampleIntervalMs))continue;
    if(!keys.every(key=>Number.isSafeInteger(a[key])&&a[key]>=0&&Number.isSafeInteger(b[key])&&b[key]>=a[key]))continue;
    for(const key of keys)deltas[key]+=b[key]-a[key];
    coverageMs+=dt;
  }
  return {deltas,seconds:coverageMs/1000,coverageMs};
}
function monitorWindowActiveRate(history, tokenKey, activeKey, start, end, sampleIntervalMs=2000) {
  const measured=monitorWindowCounters(history,[tokenKey,activeKey],start,end,sampleIntervalMs);
  const activeSeconds=measured.deltas[activeKey]/1e9;
  return {rate:activeSeconds>0?measured.deltas[tokenKey]/activeSeconds:null,activeSeconds,coverageMs:measured.coverageMs};
}
function monitorWindowGauge(history, key, start, end, sampleIntervalMs=2000) {
  let weighted=0, coverageMs=0;
  for(let i=1;i<history.length;i++) {
    const a=history[i-1], b=history[i], dt=b.at_ms-a.at_ms;
    if(a.at_ms<start||b.at_ms>end||!monitorValidPair(a,b,sampleIntervalMs))continue;
    if(!Number.isFinite(a[key])||a[key]<0||!Number.isFinite(b[key])||b[key]<0)continue;
    weighted+=(a[key]+b[key])/2*dt;
    coverageMs+=dt;
  }
  return {average:coverageMs?weighted/coverageMs:null,seconds:coverageMs/1000,coverageMs};
}
function monitorWindowMemory(history, start, end, sampleIntervalMs=2000) {
  let area=0, seconds=0, coverageMs=0;
  for(let i=1;i<history.length;i++) {
    const a=history[i-1], b=history[i], dt=b.at_ms-a.at_ms;
    if(a.at_ms<start||b.at_ms>end||!monitorValidPair(a,b,sampleIntervalMs))continue;
    const nextArea=b.process_memory_byte_seconds_total-a.process_memory_byte_seconds_total;
    const nextSeconds=b.process_memory_observed_seconds_total-a.process_memory_observed_seconds_total;
    if(!Number.isFinite(nextArea)||nextArea<0||!Number.isFinite(nextSeconds)||nextSeconds<0)continue;
    area+=nextArea;seconds+=nextSeconds;coverageMs+=dt;
  }
  return {average:seconds>0?area/seconds:null,seconds,coverageMs:seconds*1000};
}
function monitorLifetime(sample) {
  const nonnegative = v => Number.isSafeInteger(v) && v >= 0;
  const speed = (tokens, activeNs) => nonnegative(tokens) && nonnegative(activeNs) && activeNs > 0 ? tokens / (activeNs / 1e9) : null;
  const queries = sample?.cache_queries_total, hits = sample?.cache_hits_total;
  const ttftSum = sample?.ttft_ns_sum, ttftCount = sample?.ttft_count;
  const area = sample?.process_memory_byte_seconds_total, seconds = sample?.process_memory_observed_seconds_total;
  return {
    decode: speed(sample?.generation_tokens_live, sample?.decode_active_ns_total),
    prefill: speed(sample?.prefill_tokens_forwarded_live_total, sample?.prefill_active_ns_total),
    cache: nonnegative(queries) && nonnegative(hits) && queries > 0 && hits <= queries ? 100 * hits / queries : null,
    queries: nonnegative(queries) && nonnegative(hits) && hits <= queries ? queries : null,
    hits: nonnegative(queries) && nonnegative(hits) && hits <= queries ? hits : null,
    ttft: nonnegative(ttftSum) && nonnegative(ttftCount) && ttftCount > 0 && ttftSum > 0 ? ttftSum / ttftCount / 1e6 : null,
    ttftCount: nonnegative(ttftSum) && nonnegative(ttftCount) && ((ttftSum===0)===(ttftCount===0)) ? ttftCount : null,
    memory: Number.isFinite(area) && area >= 0 && Number.isFinite(seconds) && seconds > 0 ? area / seconds : null,
    memorySeconds: Number.isFinite(seconds) && seconds >= 0 ? seconds : null,
    decodeActiveNs: nonnegative(sample?.decode_active_ns_total) ? sample.decode_active_ns_total : null,
    prefillActiveNs: nonnegative(sample?.prefill_active_ns_total) ? sample.prefill_active_ns_total : null,
  };
}
function monitorWindowTTFT(history, start, end, sampleIntervalMs=2000) {
  let sum=0, count=0, coverageMs=0, tainted=false, untrusted=false;
  for(let i=1;i<history.length;i++) {
    const a=history[i-1], b=history[i], dt=b.at_ms-a.at_ms;
    if(!monitorValidPair(a,b,sampleIntervalMs)||![a.ttft_ns_sum,b.ttft_ns_sum,a.ttft_count,b.ttft_count].every(v=>Number.isSafeInteger(v)&&v>=0)){tainted=false;continue;}
    const nextSum=b.ttft_ns_sum-a.ttft_ns_sum, nextCount=b.ttft_count-a.ttft_count;
    if(nextSum<0||nextCount<0){tainted=false;continue;}
    const inWindow=a.at_ms>=start&&b.at_ms<=end;
    if((nextSum===0)!==(nextCount===0)){tainted=true;if(inWindow)untrusted=true;continue;}
    if(tainted){tainted=false;if(inWindow)untrusted=true;continue;}
    if(!inWindow)continue;
    sum+=nextSum;count+=nextCount;coverageMs+=dt;
  }
  const average=count>0?sum/count/1e6:null;
  return untrusted?{average:null,count:0,coverageMs:0}:{average:Number.isFinite(average)&&average>0?average:null,count,coverageMs};
}
function monitorIntervalMeans(history, sumKey, countKey, divisor=1, sampleIntervalMs=2000) {
  let tainted=false;
  return history.map((s,i)=>{
    const p=history[i-1], dt=p?s.at_ms-p.at_ms:0;
    const valid=p&&monitorValidPair(p,s,sampleIntervalMs)&&[p[sumKey],s[sumKey],p[countKey],s[countKey]].every(v=>Number.isSafeInteger(v)&&v>=0);
    const sum=valid?s[sumKey]-p[sumKey]:0, count=valid?s[countKey]-p[countKey]:0;
    if(!valid||sum<0||count<0){tainted=false;return {t:s.at_ms,value:null};}
    if((sum===0)!==(count===0)){tainted=true;return {t:s.at_ms,value:null};}
    if(tainted){tainted=false;return {t:s.at_ms,value:null};}
    return {t:s.at_ms,value:sum>0&&count>0?sum/count/divisor:null};
  });
}
function monitorGaugeSeries(history, key, sampleIntervalMs=2000) {
  const points=[];
  history.forEach((s,i)=>{
    const value=Number.isFinite(s[key])&&s[key]>=0?s[key]:null;
    const p=history[i-1], dt=p?s.at_ms-p.at_ms:0;
    if(value!=null&&points.at(-1)?.value!=null&&p&&!monitorValidPair(p,s,sampleIntervalMs))points.push({t:s.at_ms,value:null});
    points.push({t:s.at_ms,value});
  });
  return points;
}
function monitorSeries(history, key, derivative, sampleIntervalMs=2000) {
  return history.map((s,i) => {
    let value = typeof s[key] === 'number' ? s[key] : null;
    if (derivative) { const p=history[i-1], dt=p ? (s.at_ms-p.at_ms)/1000 : 0; value = p && monitorValidPair(p,s,sampleIntervalMs) && value != null && p[key] != null && value >= p[key] ? (value-p[key])/dt : null; }
    return { t:s.at_ms, value };
  });
}
function monitorNearestPoint(series, start, end, x, y, max) {
  if (!(end > start)) return null;
  let best = null, distance = Infinity;
  series.forEach((s, seriesIndex) => s.points.forEach((point, pointIndex) => {
    if (!Number.isFinite(point.t) || !Number.isFinite(point.value) || point.t < start || point.t > end) return;
    const px = 30 + (point.t - start) / (end - start) * 560;
    const py = 108 - point.value / max * 90;
    const d = Math.abs(px - x) + (y == null ? 0 : Math.abs(py - y) * .35);
    if (d < distance) { distance = d; best = { seriesIndex, pointIndex, point, x: px, y: py }; }
  }));
  return distance <= 28 ? best : null;
}
function monitorEscape(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function monitorFormatTTFT(valueMs) {
  if (!Number.isFinite(valueMs)) return '—';
  return valueMs>=1000?(valueMs/1000).toLocaleString(undefined,{maximumFractionDigits:1})+' s':valueMs.toLocaleString(undefined,{maximumFractionDigits:1})+' ms';
}
function monitorChartTimeLabel(at, start, end, tooltip=false) {
  const a=new Date(start), b=new Date(end);
  const sameDay=a.getFullYear()===b.getFullYear()&&a.getMonth()===b.getMonth()&&a.getDate()===b.getDate();
  const options={hour:'numeric',minute:'2-digit'};
  if(!sameDay){options.month='short';options.day='numeric';if(a.getFullYear()!==b.getFullYear())options.year='numeric';}
  if(tooltip)options.second='2-digit';
  return new Date(at).toLocaleString(undefined,options);
}
if (typeof globalThis !== 'undefined') globalThis.__mlxPanel = { computeRates, panelAt, monitorWindow, monitorHistory, monitorValidPair, monitorWindowCounters, monitorWindowActiveRate, monitorWindowGauge, monitorWindowMemory, monitorLifetime, monitorWindowTTFT, monitorIntervalMeans, monitorGaugeSeries, monitorSeries, monitorNearestPoint, monitorEscape, monitorFormatTTFT, monitorChartTimeLabel };

if (typeof document !== 'undefined') (function () {
  const mount = document.getElementById('mlx-metrics');
  if (!mount) return;
  const I18N = window.mlxI18n, t = (s,p) => I18N ? I18N.t(s,p) : s;
  const esc=monitorEscape, $=id=>document.getElementById(id);
  const liveSamples=[];
  let data=null, models=[], props=null, paused=false, windowMs=300000, selected='', lastReceived=0, state='Connecting', timer=null, inFlight=false;
  const fmt=(v,d=1)=>typeof v==='number' && Number.isFinite(v) ? v.toLocaleString(undefined,{maximumFractionDigits:d}) : '—';
  const bytes=v=>v==null?'—':v>=1073741824?fmt(v/1073741824)+' GiB':fmt(v/1048576)+' MiB';
  const ms=v=>v==null?'—':v>=1000?fmt(v/1000)+' s':fmt(v,0)+' ms';
  const duration=v=>v>=86400000?fmt(v/86400000,1)+'d':v>=3600000?fmt(v/3600000,1)+'h':v>=60000?fmt(v/60000,1)+'m':v>=1000?fmt(v/1000,1)+'s':fmt(v,0)+'ms';
  const measured=(amount,total)=>t('Measured %@ of %@',[amount,total]);
  const coverage=(value,shared,interval)=>shared-value>interval*1.5?' · '+measured(duration(value),duration(shared)):'';
  const requests=(count,kind)=>t('%@ '+kind+(count===1?' request':' requests'),[fmt(count,0)]);
  const text=(id,v)=>{const e=$(id);if(e)e.textContent=v;};
  const labelKeys=new Map();
  const label=s=>{const value=t(s);labelKeys.set(value,s);return esc(value);};
  const tile=(name,id,unit='',hint='')=>`<div class="monitor-tile"><div class="monitor-label">${label(name)}${hint?` <span class="monitor-help" tabindex="0" title="${label(hint)}" aria-label="${label(hint)}">ⓘ</span>`:''}</div><div class="monitor-value" id="${id}">—</div><div class="monitor-caption" id="${id}-caption">${label(unit)}</div></div>`;
  mount.innerHTML=`<div class="monitor-heading"><div><div class="monitor-eyebrow">MLX SERVE / ${label('Observability')}</div><h1>${label('Monitor')}</h1><p>${label('Live server health and request performance')}</p></div><span id="m-status" role="status">${label('Connecting')}</span></div>
  <div class="monitor-toolbar"><label>${label('Time range')} <select id="m-window" aria-label="${label('Time range')}"><option value="60000">1m</option><option value="300000" selected>5m</option><option value="900000">15m</option><option value="1800000">30m</option><option value="3600000">1h</option><option value="10800000">3h</option><option value="21600000">6h</option><option value="43200000">12h</option><option value="86400000">24h</option><option value="startup">${label('Since startup')}</option></select></label><label>${label('Model')} <select id="m-model" aria-label="${label('Model')}"><option value="">${label('All models')}</option></select></label><span class="monitor-spacer"></span><button id="m-pause">${label('Pause')}</button><button id="m-copy">${label('Copy diagnostics')}</button></div>
  <div id="m-notice" class="monitor-notice" role="status"></div><div id="m-meta" class="monitor-meta"></div>
  <div class="monitor-summary"><div><span class="monitor-label">${label('Loaded model')}</span><strong id="m-loaded-model">—</strong></div><div><span class="monitor-label">${label('Current phase')}</span><strong id="m-phase">—</strong></div><div><span class="monitor-label">${label('Active / queued')}</span><strong id="m-active-queued">—</strong></div></div>
  <div id="m-overview-scope" class="monitor-overview-scope monitor-meta"></div>
  <div class="monitor-grid monitor-overview">${tile('Decode speed','m-live-decode','tok/s')}${tile('Prefill speed','m-live-prefill','tok/s')}${tile('Cache hit rate','m-window-cache','Selected period','Includes finished inference requests, even if cancelled or failed.')}${tile('TTFT','m-ttft','ms')}${tile('Process memory','m-memory','GiB')}</div>
  <div class="monitor-section-head"><h2>${label('Performance over time')}</h2><span>${label('Global · hover for a sampled value')}</span></div>
  <div class="monitor-charts monitor-primary-charts">${['prefill','decode','ttft','memory'].map(id=>`<section class="monitor-chart"><div class="monitor-section-head"><h3 id="m-${id}-title">${label(({prefill:'Prefill · tok/s',decode:'Decode · tok/s',ttft:'TTFT mean · ms',memory:'Process memory · GiB'})[id])}</h3><span id="m-${id}-legend"></span></div><div class="monitor-plot"><svg id="m-chart-${id}" viewBox="0 0 600 130" role="img" tabindex="0" aria-label="${label(({prefill:'Prefill · tok/s',decode:'Decode · tok/s',ttft:'TTFT mean · ms',memory:'Process memory · GiB'})[id])}"></svg><div id="m-${id}-tooltip" class="monitor-tooltip" role="status" hidden></div></div><div class="monitor-axis"><span id="m-${id}-start"></span><span id="m-${id}-end"></span></div></section>`).join('')}</div>
  <section class="monitor-section"><div class="monitor-section-head"><h2>${label('Active requests')}</h2><span>${label('Current · model filter')}</span></div><div id="m-active" class="monitor-table-wrap"></div></section>
  <section class="monitor-section"><div class="monitor-section-head"><h2>${label('Recent requests and failures')}</h2><span>${label('Retained rows · selected period · click to inspect')}</span></div><div id="m-recent" class="monitor-table-wrap"></div><details id="m-inspector" hidden><summary>${label('Request details')}</summary><pre id="m-request-detail"></pre></details></section>
  <details id="m-details" class="monitor-section monitor-details"><summary>${label('More details')}</summary><div class="monitor-detail-body"><div id="m-coverage" class="monitor-meta"></div>
  <div class="monitor-grid monitor-latency">${tile('TTFT · p50 / p95','m-ttft-range','Successful retained requests')}${tile('Queue wait · p50 / p95','m-queue-latency','Successful retained requests')}${tile('End-to-end · p50 / p95','m-e2e','Successful retained requests')}</div>
  <div class="monitor-charts">${['latency','queue'].map(id=>`<section class="monitor-chart"><div class="monitor-section-head"><h3>${label(({latency:'Request latency',queue:'Queue and concurrency'})[id])}</h3><span id="m-${id}-legend"></span></div><div class="monitor-plot"><svg id="m-chart-${id}" viewBox="0 0 600 130" role="img" tabindex="0" aria-label="${label(({latency:'Request latency',queue:'Queue and concurrency'})[id])}"></svg><div id="m-${id}-tooltip" class="monitor-tooltip" role="status" hidden></div></div><div class="monitor-axis"><span id="m-${id}-start"></span><span id="m-${id}-end"></span></div></section>`).join('')}</div>
  <section class="monitor-section"><div class="monitor-section-head"><h2>${label('Model inventory')}</h2><span>${label('Current · disk and RAM are separate')}</span></div><div id="m-inventory" class="monitor-table-wrap"></div></section>
  <div class="monitor-pair"><section class="monitor-section"><div class="monitor-section-head"><h2>${label('Resources')}</h2><span>${label('Current · global')}</span></div><div id="m-resources" class="monitor-facts"></div></section><section class="monitor-section"><div class="monitor-section-head"><h2>${label('Prefix cache')}</h2><span>${label('Lifetime · global')}</span></div><div id="m-cache" class="monitor-facts"></div></section></div>
  <section class="monitor-section"><div class="monitor-section-head"><h2>${label('Per-model performance')}</h2><span>${label('Retained · selected period')}</span></div><div id="m-model-stats" class="monitor-table-wrap"></div></section>
  <section class="monitor-section"><div class="monitor-section-head"><h2>${label('Events')}</h2><span>${label('Retained rows · current server instance')}</span></div><div id="m-events" class="monitor-events"></div></section>
  <details class="monitor-section"><summary>${label('Runtime diagnostics')}</summary><div id="m-diagnostics" class="monitor-facts"></div></details></div></details>`;

  const staticLabels=[];
  const walker=document.createTreeWalker(mount,NodeFilter.SHOW_TEXT);
  while(walker.nextNode()){const node=walker.currentNode,key=node.textContent.trim();if(key)staticLabels.push([node,labelKeys.get(key)||key]);}

  function facts(id, entries) { $(id).innerHTML=entries.filter(([,v])=>v!=null).map(([k,v])=>`<div><span>${label(k)}</span><strong>${esc(v)}</strong></div>`).join(''); }
  function table(id, headings, rows, empty) { const html=`<table aria-label="${esc(t(({ 'm-active':'Active requests','m-recent':'Recent requests','m-inventory':'Model inventory','m-model-stats':'Per-model performance'})[id]||id))}"><thead><tr>${headings.map(h=>`<th scope="col">${label(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${headings.length}" class="monitor-empty">${label(empty)}</td></tr>`}</tbody></table>`; if($(id).innerHTML!==html)$(id).innerHTML=html; }
  const chartData=new Map(), chartFocus=new Map(), chartColors=['#76a9fa','#63d7b0','#d9a5fc'];
  function showChartFocus(id) {
    const svg=$('m-chart-'+id), tip=$('m-'+id+'-tooltip'), info=chartData.get(id), focus=chartFocus.get(id);
    if(!info||!focus||!svg.getBoundingClientRect().width){tip.hidden=true;return;}
    const rect=svg.getBoundingClientRect(), matrix=svg.getScreenCTM();
    let x=focus.x, y=focus.y;
    if(focus.clientX!=null&&matrix){const point=svg.createSVGPoint();point.x=focus.clientX;point.y=focus.clientY;const local=point.matrixTransform(matrix.inverse());x=local.x;y=local.y;}
    const saved=focus.clientX!=null||focus.seriesIndex==null?null:info.series[focus.seriesIndex]?.points.find(p=>p.t===focus.time&&Number.isFinite(p.value));
    const hit=saved?{seriesIndex:focus.seriesIndex,point:saved,x:30+(saved.t-info.start)/(info.end-info.start)*560,y:108-saved.value/info.max*90}:monitorNearestPoint(info.series,info.start,info.end,x,y,info.max);
    if(focus.clientX!=null){focus.seriesIndex=hit?.seriesIndex;focus.time=hit?.point.t;}
    const mark=svg.querySelector('.monitor-chart-focus');
    if(!hit){mark.innerHTML='';tip.textContent=t('No measured sample here');tip.hidden=false;}
    else {
      const color=chartColors[hit.seriesIndex], name=t(info.series[hit.seriesIndex].name);
      mark.innerHTML=`<line x1="${hit.x}" y1="18" x2="${hit.x}" y2="108" stroke="var(--fg)" stroke-opacity=".5" stroke-dasharray="3 3"/><circle cx="${hit.x}" cy="${hit.y}" r="5" fill="${color}" stroke="var(--card)" stroke-width="2"/>`;
      tip.innerHTML=`<strong>${esc(name)} · ${esc(info.formatValue(hit.point.value))} ${esc(info.unit)}</strong><span>${esc(monitorChartTimeLabel(hit.point.t,info.start,info.end,true))}</span>`;
      tip.hidden=false;
    }
    const plot=svg.parentElement, plotRect=plot.getBoundingClientRect(), point=svg.createSVGPoint();point.x=hit?.x??x;point.y=hit?.y??y;
    const screen=matrix?point.matrixTransform(matrix):{x:rect.left+x/600*rect.width,y:rect.top+y/130*rect.height};
    tip.style.left=Math.max(4,Math.min(screen.x-plotRect.left+10,plot.clientWidth-tip.offsetWidth-4))+'px';
    tip.style.top=Math.max(4,Math.min(screen.y-plotRect.top+8,plot.clientHeight-tip.offsetHeight-4))+'px';
  }
  function chart(id, series, start, end, unit, digits=1, formatValue=v=>fmt(v,digits)) {
    const svg=$('m-chart-'+id);
    const values=series.flatMap(s=>s.points.map(p=>p.value)).filter(v=>v!=null&&Number.isFinite(v));
    const max=Math.max(1,...values), x=t=>30+((t-start)/(end-start))*560, y=v=>108-(v/max)*90;
    let out=`<line x1="30" y1="108" x2="590" y2="108" stroke="var(--line)"/><text x="0" y="16" fill="var(--dim)" font-size="11">${esc(id==='ttft'?formatValue(max)+' '+unit:fmt(max))}</text>`;
    series.forEach((s,i)=>{let segment=[];const flush=()=>{if(segment.length>1)out+=`<polyline points="${segment.join(' ')}" fill="none" stroke="${chartColors[i]}" stroke-width="2"/>`;else if(segment.length)out+=`<circle cx="${segment[0].split(',')[0]}" cy="${segment[0].split(',')[1]}" r="2" fill="${chartColors[i]}"/>`;segment=[];};s.points.forEach(p=>{if(!Number.isFinite(p.value)){flush();return;}segment.push(x(p.t).toFixed(1)+','+y(p.value).toFixed(1));});flush();});
    if(!values.length)out+=`<text x="300" y="65" text-anchor="middle" fill="var(--dim)" font-size="13">${label('No samples in this period')}</text>`;
    svg.innerHTML=out+'<g class="monitor-chart-focus" aria-hidden="true"></g>';
    chartData.set(id,{series,start,end,max,unit,digits,formatValue});
    text('m-'+id+'-legend',series.map(s=>t(s.name)+' '+formatValue(s.points.filter(p=>Number.isFinite(p.value)).at(-1)?.value)).join(' / ')+' · '+unit);
    text('m-'+id+'-start',monitorChartTimeLabel(start,start,end));text('m-'+id+'-end',monitorChartTimeLabel(end,start,end));
    if(chartFocus.has(id))showChartFocus(id);
  }
  for(const id of ['prefill','decode','ttft','memory','latency','queue']){
    const svg=$('m-chart-'+id), tip=$('m-'+id+'-tooltip');
    svg.addEventListener('pointermove',e=>{chartFocus.set(id,{clientX:e.clientX,clientY:e.clientY});showChartFocus(id);});
    svg.addEventListener('pointerleave',()=>{chartFocus.delete(id);tip.hidden=true;svg.querySelector('.monitor-chart-focus').innerHTML='';});
    svg.addEventListener('pointerdown',e=>{chartFocus.set(id,{clientX:e.clientX,clientY:e.clientY});showChartFocus(id);});
    svg.addEventListener('focus',()=>{if(chartFocus.has(id))return;const info=chartData.get(id);if(!info)return;const entries=info.series.flatMap((s,seriesIndex)=>s.points.filter(p=>Number.isFinite(p.value)).map(point=>({point,seriesIndex}))).sort((a,b)=>a.point.t-b.point.t||a.seriesIndex-b.seriesIndex);const entry=entries.at(-1);if(entry){chartFocus.set(id,{seriesIndex:entry.seriesIndex,time:entry.point.t,x:30+(entry.point.t-info.start)/(info.end-info.start)*560,y:108-entry.point.value/info.max*90});showChartFocus(id);}});
    svg.addEventListener('blur',()=>{chartFocus.delete(id);tip.hidden=true;svg.querySelector('.monitor-chart-focus').innerHTML='';});
    svg.addEventListener('keydown',e=>{if(e.key==='Escape'){chartFocus.delete(id);tip.hidden=true;svg.querySelector('.monitor-chart-focus').innerHTML='';e.preventDefault();return;}if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const info=chartData.get(id);if(!info)return;const entries=info.series.flatMap((s,seriesIndex)=>s.points.filter(p=>Number.isFinite(p.value)).map(point=>({point,seriesIndex}))).sort((a,b)=>a.point.t-b.point.t||a.seriesIndex-b.seriesIndex);if(!entries.length)return;e.preventDefault();const focus=chartFocus.get(id), current=entries.findIndex(e=>e.seriesIndex===focus?.seriesIndex&&e.point.t===focus.time);const index=e.key==='Home'?0:e.key==='End'?entries.length-1:e.key==='ArrowLeft'?Math.max(0,current<0?entries.length-1:current-1):Math.min(entries.length-1,current+1);const {point,seriesIndex}=entries[index];chartFocus.set(id,{seriesIndex,time:point.t,x:30+(point.t-info.start)/(info.end-info.start)*560,y:108-point.value/info.max*90});showChartFocus(id);});
  }
  function render() {
    if(document.hidden || (location.hash && location.hash!=='#monitor'))return;
    const m=data?.monitor||{}, g=data?.gauges||{}, c=data?.counters||{}, now=m.server?.sampled_at_ms||Date.now(), w=monitorWindow(data||{},now,windowMs,selected), res=m.resources||(props?.memory?{mlx_active_bytes:props.memory.active_bytes,mlx_cache_bytes:props.memory.cache_bytes,system_available_bytes:props.memory.available_bytes}:{}), diag=m.diagnostics||{}, cache=m.cache||{};
    const age=lastReceived?Math.max(Date.now()-lastReceived,m.server?.sampled_at_ms?Date.now()-m.server.sampled_at_ms:0):0;
    text('m-status',t(paused?'Paused':state==='Live'&&age>6500?'Stale':state)); $('m-status').dataset.state=paused?'paused':state.toLowerCase();
    text('m-notice',state==='Metrics disabled'?t('Metrics are disabled. Restart with --metrics to collect request and resource history.'):state==='Offline'?t(data?'Server unavailable. Last received values are retained; they are not live.':'Server unavailable.'):state==='Unauthorized'?t('Authentication required. Open the console with an API key.'):state==='Live'&&age>6500?t('Telemetry is stale. Showing the last received sample.'):t('Request records contain metadata only. Unavailable measurements are shown as —.'));
    text('m-meta',[m.server?.version?'v'+m.server.version:'',m.server?.uptime_seconds!=null?t('Uptime')+' '+fmt(m.server.uptime_seconds/60,0)+'m':'',lastReceived?t('Last sample')+' '+new Date(m.server?.sampled_at_ms||lastReceived).toLocaleTimeString():''].filter(Boolean).join(' · '));
    const inventory=m.models||models, active=(m.active_requests||[]).filter(r=>!selected||r.model===selected);
    const ready=inventory.filter(r=>r.state==='ready'&&(!selected||r.id===selected)).map(r=>r.id);
    text('m-loaded-model',ready.length?ready.join(', '):t('None loaded'));
    text('m-phase',active.length?[...new Set(active.map(r=>t(r.phase||'running')))].join(' · '):t('Idle'));
    text('m-active-queued',(data?fmt(g.requests_running,0):'—')+' / '+(data?fmt(g.requests_waiting,0):'—'));
    const startup=windowMs==='startup', started=m.server?.started_at_ms, validStart=Number.isFinite(started)&&started<=now;
    const start=startup?(validStart&&started<now?started:now-1):now-windowMs, interval=m.server?.sample_interval_ms||2000, history=monitorHistory(m);
    const decode=monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',start,now,interval);
    const prefill=monitorWindowActiveRate(history,'prefill_tokens_forwarded_live_total','prefill_active_ns_total',start,now,interval);
    const cacheWindow=monitorWindowCounters(history,['cache_queries_total','cache_hits_total'],start,now,interval);
    const ttft=monitorWindowTTFT(history,start,now,interval);
    const memory=history.some(s=>Number.isFinite(s.process_memory_observed_seconds_total))?monitorWindowMemory(history,start,now,interval):monitorWindowGauge(history,'process_bytes',start,now,interval);
    const sharedCoverage=monitorWindowCounters(history,[],start,now,interval).coverageMs;
    const lifetime=startup&&validStart&&m.lifetime_totals?monitorLifetime(m.lifetime_totals):null;
    const windowLabel=$('m-window').selectedOptions[0]?.textContent||duration(windowMs);
    text('m-overview-scope',startup?t('Global')+' · '+windowLabel+(validStart?' · '+t('Runtime')+' '+duration(now-started):' · '+t('Start time unavailable')):t('Global')+' · '+measured(duration(sharedCoverage),windowLabel));
    text('m-live-decode',startup?(lifetime?.decode==null?'—':fmt(lifetime.decode)):(decode.rate==null?'—':fmt(decode.rate)));
    text('m-live-prefill',startup?(lifetime?.prefill==null?'—':fmt(lifetime.prefill)):(prefill.rate==null?'—':fmt(prefill.rate)));
    const queryCount=startup?lifetime?.queries:cacheWindow.deltas.cache_queries_total, hitCount=startup?lifetime?.hits:cacheWindow.deltas.cache_hits_total;
    text('m-window-cache',startup?(lifetime?.cache==null?'—':fmt(lifetime.cache)+'%'):(cacheWindow.seconds&&queryCount?fmt(Math.min(100,100*hitCount/queryCount))+'%':'—'));
    text('m-ttft',monitorFormatTTFT(startup?lifetime?.ttft:ttft.average));text('m-memory',bytes(startup?lifetime?.memory:memory.average));
    text('m-live-decode-caption',startup?(lifetime?.decode!=null?t('Processing average')+' · '+t('%@ active',[duration(lifetime.decodeActiveNs/1e6)]):t(lifetime?.decodeActiveNs===0?'No processing in this period':'No measured samples')):(decode.rate!=null?t('Processing average')+' · '+t('%@ active',[duration(decode.activeSeconds*1000)]):t(decode.coverageMs?'No processing in this period':'No measured samples'))+coverage(decode.coverageMs,sharedCoverage,interval));
    text('m-live-prefill-caption',startup?(lifetime?.prefill!=null?t('Processing average')+' · '+t('%@ active',[duration(lifetime.prefillActiveNs/1e6)]):t(lifetime?.prefillActiveNs===0?'No processing in this period':'No measured samples')):(prefill.rate!=null?t('Processing average')+' · '+t('%@ active',[duration(prefill.activeSeconds*1000)]):t(prefill.coverageMs?'No processing in this period':'No measured samples'))+coverage(prefill.coverageMs,sharedCoverage,interval));
    text('m-window-cache-caption',startup?(queryCount==null?t('No measured samples'):queryCount?(t(hitCount===1?'%@ hit':'%@ hits',[fmt(hitCount,0)])+' / '+requests(queryCount,'finished')):t('No finished requests')):(cacheWindow.seconds?(queryCount?t(hitCount===1?'%@ hit':'%@ hits',[fmt(hitCount,0)])+' / '+requests(queryCount,'finished'):t('No finished requests')):t('No measured samples'))+coverage(cacheWindow.coverageMs,sharedCoverage,interval));
    text('m-ttft-caption',startup?(lifetime?.ttft!=null?t('Average')+' · '+requests(lifetime.ttftCount,'successful'):t(lifetime?.ttftCount===0?'No successful requests':'No measured samples')):(ttft.coverageMs?(ttft.average!=null?t('Average')+' · '+requests(ttft.count,'successful'):t(ttft.count?'TTFT unavailable':'No successful requests')):t('No measured samples'))+coverage(ttft.coverageMs,sharedCoverage,interval));
    text('m-memory-caption',startup?(lifetime?.memorySeconds>0?t('Average')+' · '+measured(duration(lifetime.memorySeconds*1000),duration(now-started)):t('No measured samples')):(memory.seconds?t('Average'):t('No measured samples'))+coverage(memory.coverageMs,sharedCoverage,interval));
    text('m-coverage',`${(m.recent_requests||[]).length} / ${fmt(m.retention?.request_capacity??256,0)} ${t('retained requests')}${w.partial?' · '+t('Partial coverage'):''} · ${(m.events||[]).length} / ${fmt(m.retention?.event_capacity??128,0)} ${t('retained events')} · ${t('Older chart detail is compacted')} · ${t('Raw history retained')}: ${fmt((m.retention?.history_seconds||0)/60,0)}m`);
    for(const [id,key] of [['m-ttft-range','ttft_ms'],['m-queue-latency','queue_ms'],['m-e2e','e2e_ms']])text(id,ms(w.percentile(key,.5))+' / '+ms(w.percentile(key,.95)));
    chart('prefill',[{name:'Prefill',points:monitorSeries(w.history,'prefill_tokens_forwarded_live_total',true,interval)}],start,now,'tok/s',0);
    chart('decode',[{name:'Decode',points:monitorSeries(w.history,'generation_tokens_live',true,interval)}],start,now,'tok/s',1);
    const ttftPoints=monitorIntervalMeans(history,'ttft_ns_sum','ttft_count',1e6,interval).filter(p=>p.t>=start&&p.t<=now);
    const ttftSeconds=ttftPoints.some(p=>p.value>=1000);
    const ttftUnit=ttftSeconds?'s':'ms';
    const ttftTitle=t('TTFT mean')+' · '+ttftUnit;
    text('m-ttft-title',ttftTitle);$('m-chart-ttft').setAttribute('aria-label',ttftTitle);
    chart('ttft',[{name:'TTFT mean',points:ttftPoints}],start,now,ttftUnit,1,v=>v==null?'—':ttftSeconds?fmt(v/1000):fmt(v));
    chart('memory',[{name:'Process',points:monitorGaugeSeries(w.history,'process_bytes',interval).map(p=>({...p,value:p.value==null?null:p.value/1073741824}))}],start,now,'GiB · '+t('global'));
    chart('latency',[{name:'End-to-end',points:w.requests.map(r=>({t:r.finished_at_ms,value:r.e2e_ms}))},{name:'Queue wait',points:w.requests.map(r=>({t:r.finished_at_ms,value:r.queue_ms}))}],start,now,'ms');
    chart('queue',[{name:'Active',points:monitorSeries(w.history,'running')},{name:'Queued',points:monitorSeries(w.history,'queued')}],start,now,t('global'));
    table('m-active',['Request','Model','Phase','Elapsed','Queue wait','Prompt / cached / output'],active.map(r=>[`<button class="monitor-request" data-request="${esc(r.id)}">${esc(r.id)}</button>`,esc(r.model),label(r.phase),ms(now-r.started_at_ms),ms(r.queue_ms),[r.prompt_tokens,r.cached_tokens,r.output_tokens].map(v=>fmt(v,0)).join(' / ')]),'No active requests');
    table('m-recent',['Request','Model','Outcome','TTFT','End-to-end','Prompt / cached / output'],w.requests.slice().reverse().map(r=>[`<button class="monitor-request" data-request="${esc(r.id)}">${esc(r.id)}</button>`,esc(r.model),`<span class="monitor-outcome ${r.outcome==='success'?'ok':'other'}">${esc(r.outcome)}</span>${r.error_code?'<div class="monitor-caption">'+esc(r.error_code)+'</div>':''}`,ms(r.ttft_ms),ms(r.e2e_ms),[r.prompt_tokens,r.cached_tokens,r.output_tokens].map(v=>fmt(v,0)).join(' / ')]),'No completed requests in this period');
    table('m-inventory',['Model','State','Backend','Quantization','Context','Disk','RAM'],inventory.filter(r=>!selected||r.id===selected).map(r=>[esc(r.id),label(r.state||'unknown'),esc(r.backend??r.engine??'—'),esc(r.quantization??(r.quantization_bits!=null?r.quantization_bits+' bit':'—')),fmt(r.context_length,0),bytes(r.bytes_on_disk),bytes(r.bytes_resident)+(r.estimate&&r.estimated_resident_bytes!=null?'<div class="monitor-caption">'+bytes(r.estimated_resident_bytes)+' '+label('(estimate)')+'</div>':'')]),'No models discovered');
    facts('m-resources',[['CPU',res.cpu_pct==null?null:fmt(res.cpu_pct)+'%'],['GPU',res.gpu_pct==null?null:fmt(res.gpu_pct)+'%'],['Process footprint',bytes(res.process_bytes)],['MLX active',bytes(res.mlx_active_bytes)],['MLX reusable pool',bytes(res.mlx_cache_bytes)],['System available / total',bytes(res.system_available_bytes)+' / '+bytes(res.system_total_bytes)],['Memory pressure',res.memory_pressure_pct==null?null:fmt(res.memory_pressure_pct)+'%'],['Swap used',bytes(res.swap_used_bytes)]]);
    facts('m-cache',[['Request hit rate',cache.queries>0?fmt(100*cache.hits/cache.queries)+'%':'—'],['Hits / queries',fmt(cache.hits,0)+' / '+fmt(cache.queries,0)],['Reused tokens',fmt(cache.reused_tokens,0)],['Token reuse',c.prompt_tokens_total>0?fmt(100*c.prefix_cache_tokens_total/c.prompt_tokens_total)+'%':null],['Hot cache / capacity',bytes(cache.hot_bytes)+' / '+bytes(cache.capacity_bytes)],['Entries',cache.entries],['Evictions',cache.evictions]]);
    const ids=[...new Set(w.requests.map(r=>r.model))];
    table('m-model-stats',['Model','Requests','Errors','Output tokens','TTFT p50 / p95'],ids.map(id=>{const rows=w.requests.filter(r=>r.model===id), mw=monitorWindow(data||{},now,windowMs,id);return[esc(id),fmt(rows.length,0),fmt(rows.filter(r=>r.outcome==='failed'||r.outcome==='rejected').length,0),fmt(rows.reduce((n,r)=>n+(r.output_tokens||0),0),0),ms(mw.percentile('ttft_ms',.5))+' / '+ms(mw.percentile('ttft_ms',.95))];}),'No completed requests in this period');
    const events=(m.events||[]).filter(e=>e.at_ms>=start&&(!selected||!e.model||e.model===selected)).slice().reverse();
    $('m-events').innerHTML=events.length?events.map(e=>`<div><time>${esc(new Date(e.at_ms).toLocaleTimeString())}</time><strong>${esc(e.kind)}</strong><span>${esc([e.model,e.code,e.message].filter(Boolean).join(' · '))}</span></div>`).join(''):`<p class="monitor-empty">${label('No events in this period')}</p>`;
    const pre=w.requests.reduce((n,r)=>n+(r.prefill_ms||0),0), dec=w.requests.reduce((n,r)=>n+(r.decode_ms||0),0);
    facts('m-diagnostics',[['Concurrency limit',diag.max_concurrent],['Resident model limit',diag.max_resident_models],['Resident memory limit',diag.max_resident_bytes===0?t('Automatic'):bytes(diag.max_resident_bytes)],['Last batch size',g.batched_group_size],['Prefill share · retained phase time',pre+dec?fmt(pre/(pre+dec)*100)+'%':'—'],['KV quantization',diag.kv_quant],['KV attention mode',diag.kv_attn_mode],['Prefill chunk',diag.prefill_chunk===0?t('Automatic'):diag.prefill_chunk],['Speculative acceptance',diag.speculative_drafted>0?fmt(diag.speculative_accepted/diag.speculative_drafted*100)+'%':null],['ANE layers',diag.ane_layers],['ANE weight copies',bytes(diag.ane_int8_bytes)],['N-gram warm progress',bytes(diag.ngram_warm_bytes)],['Request timeout',diag.request_timeout_seconds==null?null:fmt(diag.request_timeout_seconds)+' s'],['Serial decode reasons · lifetime',Object.entries(diag.decode_serial_reasons||data?.decode_serial||{}).map(([k,v])=>k+': '+v).join(' · ')||null]]);
    const options=[...new Set(inventory.map(r=>r.id).concat((m.recent_requests||[]).map(r=>r.model)))].sort();
    const select=$('m-model');if(JSON.stringify(options)!==select.dataset.options){select.innerHTML=`<option value="">${label('All models')}</option>`+options.map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join('');select.value=selected;select.dataset.options=JSON.stringify(options);}
  }
  function headers(){const p=new URLSearchParams(location.search),key=p.get('api_key')||p.get('key');return key?{Authorization:'Bearer '+key}:{};}
  async function tick(){
    if(inFlight)return;
    if(paused||document.hidden||(location.hash&&location.hash!=='#monitor')){timer=setTimeout(tick,2000);return;}
    inFlight=true;
    try{const response=await fetch('/metrics.json',{headers:headers(),signal:AbortSignal.timeout(5000)});if(response.status===503){state='Metrics disabled';data=null;liveSamples.length=0;lastReceived=0;}else if(response.status===401||response.status===403){state='Unauthorized';}else{if(!response.ok)throw Error(response.status);const next=await response.json();if(next.monitor?.server?.instance_id!==data?.monitor?.server?.instance_id)liveSamples.length=0;data=next;state='Live';lastReceived=Date.now();const at=data.monitor?.server?.sampled_at_ms||lastReceived;if(liveSamples.at(-1)?.t!==at){liveSamples.push({t:at,live:data.gauges?.generation_tokens_live||0,pre:data.gauges?.prefill_tokens_live||0,req:data.counters?.requests_success_total||0});while(liveSamples.length>120||liveSamples[0]?.t<at-120000)liveSamples.shift();}}}
    catch(e){state='Offline';}
    finally{inFlight=false;render();timer=setTimeout(tick,2000);}
  }
  window.mlxMonitor={updateModels(value){models=value;render();},updateProps(value){props=value;render();}};
  $('m-window').addEventListener('change',e=>{windowMs=e.target.value==='startup'?'startup':Number(e.target.value);render();});
  $('m-model').addEventListener('change',e=>{selected=e.target.value;render();});
  $('m-pause').addEventListener('click',()=>{paused=!paused;text('m-pause',t(paused?'Resume':'Pause'));render();if(!paused&&!inFlight){clearTimeout(timer);tick();}});
  $('m-copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(JSON.stringify({captured_at:new Date().toISOString(),status:state,selected_model:selected,window_ms:windowMs,monitor:data?.monitor||{models,props}},null,2));text('m-copy',t('Copied'));}catch(e){text('m-copy',t('Copy failed'));}setTimeout(()=>text('m-copy',t('Copy diagnostics')),1800);});
  mount.addEventListener('click',e=>{const button=e.target.closest('[data-request]');if(!button)return;const row=[...(data?.monitor?.recent_requests||[]),...(data?.monitor?.active_requests||[])].find(r=>String(r.id)===button.dataset.request);if(row){$('m-inspector').hidden=false;$('m-inspector').open=true;text('m-request-detail',JSON.stringify(row,null,2));}});
  window.addEventListener('hashchange',()=>{render();if(!inFlight){clearTimeout(timer);tick();}});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){render();if(!inFlight){clearTimeout(timer);tick();}}});
  if(I18N)I18N.onChange(()=>{ staticLabels.forEach(([node,key])=>{node.textContent=t(key);});render(); });
  render();tick();
})();
