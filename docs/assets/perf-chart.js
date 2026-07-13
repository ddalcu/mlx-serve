// Interactive performance chart — shared by index.html and
// speculative-decoding/. Expects #perfChart, #chartPills, #chartLegend,
// #chartTooltip in the DOM; no-ops if the chart isn't on the page.
(function () {
  if (!document.getElementById('perfChart')) return;

  // ─── Performance chart (real data, mlx-serve v26.7.6 vs LM Studio 0.4.15, M4 Max 128GB) ───
  // Source: docs/perf-csvs/{gemma,qwen36}-26.7.6.csv (decode_tps per workload).
  // Workloads: [Decode, Echo, Code] in tokens/sec.
  const PERF = {
    'gemma-e2b':   { name: 'Gemma 4 E2B', series: {
      'LM Studio': [194.0, 192.8, 190.6], 'mlx-serve': [198.6, 194.2, 190.6],
      '+ PLD': [188.6, 407.0, 191.6], '+ Drafter': [172.6, 233.1, 239.2] } },
    'gemma-e4b':   { name: 'Gemma 4 E4B', series: {
      'LM Studio': [120.2, 118.7, 118.8], 'mlx-serve': [120.1, 118.0, 117.9],
      '+ PLD': [116.4, 252.4, 119.9], '+ Drafter': [113.9, 228.1, 194.3] } },
    'gemma-31b':   { name: 'Gemma 4 31B', series: {
      'LM Studio': [25.6, 25.6, 25.1], 'mlx-serve': [25.4, 25.3, 25.0],
      '+ PLD': [25.1, 50.6, 28.7], '+ Drafter': [23.8, 37.8, 31.1] } },
    'gemma-26moe': { name: 'Gemma 4 26B-A4B MoE', series: {
      'LM Studio': [118.3, 118.1, 117.5], 'mlx-serve': [118.1, 115.9, 115.7],
      '+ PLD': [114.1, 191.7, 123.8] } },
    'qwen-27b':    { name: 'Qwen 3.6 27B', series: {
      'LM Studio': [30.4, 29.9, 29.5], 'mlx-serve': [29.2, 28.9, 29.0],
      '+ PLD': [28.8, 63.5, 30.3], '+ MTP': [40.1, 58.7, 58.4] } },
    'qwen-35moe':  { name: 'Qwen 3.6 35B-A3B MoE', series: {
      'LM Studio': [103.2, 103.0, 103.0], 'mlx-serve': [131.3, 130.0, 130.7],
      '+ PLD': [125.3, 238.8, 140.4], '+ MTP': [135.9, 185.6, 175.1] } },
  };
  const WORKLOADS = ['Decode', 'Echo', 'Code'];
  const COLORS = {
    'LM Studio': '#c2c2cc', 'mlx-serve': '#0071e3',
    '+ PLD': '#30d158', '+ Drafter': '#f7a41d', '+ MTP': '#bf5af2'
  };

  const canvas = document.getElementById('perfChart');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('chartTooltip');
  // Default pill: the 35B-A3B MoE is the strongest showing — +27% raw decode
  // over LM Studio before any speculation, 2.3x echo with PLD, +70% code with MTP.
  let activeKey = 'qwen-35moe';
  let animProgress = 0, animRAF = null;
  let hitRects = [];

  function niceMax(v) {
    const steps = [10, 20, 25, 40, 50, 80, 100, 120, 150, 200, 250, 300];
    for (const s of steps) if (v <= s) return s;
    return Math.ceil(v / 100) * 100;
  }

  function roundedTopRect(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, Math.abs(h));
    c.beginPath();
    c.moveTo(x, y + h);
    c.lineTo(x, y + r);
    c.arcTo(x, y, x + r, y, r);
    c.lineTo(x + w - r, y);
    c.arcTo(x + w, y, x + w, y + r, r);
    c.lineTo(x + w, y + h);
    c.closePath();
  }

  function drawChart() {
    const dpr = window.devicePixelRatio || 1;
    const wrap = canvas.parentElement;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const data = PERF[activeKey];
    const seriesNames = Object.keys(data.series);
    const padL = 46, padR = 14, padT = 16, padB = 40;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const baseY = padT + plotH;

    let maxVal = 0;
    seriesNames.forEach(s => data.series[s].forEach(v => { if (v > maxVal) maxVal = v; }));
    const top = niceMax(maxVal);

    // grid + y labels
    ctx.font = '500 11px ' + getComputedStyle(document.body).getPropertyValue('--mono');
    ctx.textBaseline = 'middle';
    const lines = 4;
    for (let i = 0; i <= lines; i++) {
      const val = top * i / lines;
      const y = baseY - plotH * i / lines;
      ctx.strokeStyle = 'rgba(0,0,0,0.06)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, y + 0.5); ctx.lineTo(W - padR, y + 0.5); ctx.stroke();
      ctx.fillStyle = '#86868b'; ctx.textAlign = 'right';
      ctx.fillText(Math.round(val), padL - 10, y);
    }

    const groupW = plotW / WORKLOADS.length;
    const innerPad = groupW * 0.16;
    const usable = groupW - innerPad * 2;
    const gap = 6;
    const barW = (usable - gap * (seriesNames.length - 1)) / seriesNames.length;

    hitRects = [];
    ctx.textAlign = 'center';

    WORKLOADS.forEach((wl, wi) => {
      const gx = padL + groupW * wi + innerPad;
      seriesNames.forEach((s, si) => {
        const val = data.series[s][wi];
        const fullH = plotH * (val / top);
        const h = fullH * animProgress;
        const x = gx + si * (barW + gap);
        const y = baseY - h;
        ctx.fillStyle = COLORS[s] || '#999';
        roundedTopRect(ctx, x, y, barW, h, 4);
        ctx.fill();
        hitRects.push({ x, y: baseY - fullH, w: barW, h: fullH, series: s, workload: wl, value: val });
      });
      // workload label
      ctx.fillStyle = '#1d1d1f';
      ctx.font = '600 12px -apple-system, sans-serif';
      ctx.textBaseline = 'top';
      ctx.fillText(wl, padL + groupW * wi + groupW / 2, baseY + 12);
      ctx.textBaseline = 'middle';
    });
  }

  function runAnim() {
    cancelAnimationFrame(animRAF);
    animProgress = 0;
    const start = performance.now(), dur = 750;
    function step(now) {
      const p = Math.min(1, (now - start) / dur);
      animProgress = 1 - Math.pow(1 - p, 3);
      drawChart();
      if (p < 1) animRAF = requestAnimationFrame(step);
      else { animProgress = 1; drawChart(); }
    }
    animRAF = requestAnimationFrame(step);
  }

  function buildPills() {
    const wrap = document.getElementById('chartPills');
    wrap.innerHTML = '';
    Object.keys(PERF).forEach(key => {
      const b = document.createElement('button');
      b.className = 'chart-pill' + (key === activeKey ? ' active' : '');
      b.textContent = PERF[key].name;
      b.onclick = () => {
        activeKey = key;
        wrap.querySelectorAll('.chart-pill').forEach(p => p.classList.remove('active'));
        b.classList.add('active');
        buildLegend();
        runAnim();
      };
      wrap.appendChild(b);
    });
  }

  function buildLegend() {
    const wrap = document.getElementById('chartLegend');
    wrap.innerHTML = '';
    Object.keys(PERF[activeKey].series).forEach(s => {
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = '<span class="legend-swatch" style="background:' + (COLORS[s] || '#999') + '"></span>' + s;
      wrap.appendChild(item);
    });
  }

  // hover tooltip
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let hit = null;
    for (const b of hitRects) {
      if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) { hit = b; break; }
    }
    if (hit) {
      const base = PERF[activeKey].series['LM Studio'][WORKLOADS.indexOf(hit.workload)];
      let delta = '';
      if (hit.series !== 'LM Studio' && base) {
        const pct = Math.round((hit.value / base - 1) * 100);
        if (pct > 0) delta = '<div class="tt-delta">+' + pct + '% vs LM Studio</div>';
      }
      tooltip.innerHTML = '<div class="tt-series">' + hit.series + ' · ' + hit.workload + '</div>' +
        '<div class="tt-val">' + hit.value.toFixed(1) + ' tok/s</div>' + delta;
      tooltip.style.left = (hit.x + hit.w / 2) + 'px';
      tooltip.style.top = hit.y + 'px';
      tooltip.style.opacity = '1';
    } else {
      tooltip.style.opacity = '0';
    }
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });

  // init + animate when scrolled into view
  buildPills();
  buildLegend();
  drawChart();
  let chartShown = false;
  const chartObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !chartShown) { chartShown = true; runAnim(); }
    });
  }, { threshold: 0.25 });
  chartObserver.observe(canvas.parentElement);

  let resizeTO;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTO);
    resizeTO = setTimeout(drawChart, 120);
  });

})();
