import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const htmlDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'html');
const source = name => readFileSync(join(htmlDir, name), 'utf8');
const slots = [
  'fixture',
  source('theme.js') + '\n;\n' + source('i18n.js'),
  source('app.css'),
  'fixture',
  `<div id="mlx-metrics"></div><script>${source('metrics.js')}</script>`,
  '11267',
  source('app.js'),
];
let slot = 0;
const page = source('index.html').replace(/\{[sd]\}/g, () => slots[slot++]);
if (slot !== slots.length) throw new Error(`Expected ${slots.length} HTML slots, got ${slot}`);

let polls = 0;
function snapshot() {
  const now = Date.now();
  const start = now - 300_000;
  const started = now - 30 * 3_600_000;
  const ttftMs = Number(process.env.MONITOR_FIXTURE_TTFT_MS || 101_403.9);
  const memoryGap = process.env.MONITOR_FIXTURE_GAP === '1';
  const baseMemorySeconds = (start - started) / 1000;
  const archived = Array.from({ length: 6 }, (_, i) => ({
    at_ms: i === 5 ? start - 5000 : started + i * 6 * 3_600_000,
    continuity_id: 1,
    generation_tokens_live: 0, prefill_tokens_forwarded_live_total: 0,
    decode_active_ns_total: 0, prefill_active_ns_total: 0,
    cache_queries_total: 1000, cache_hits_total: 900,
    ttft_count: 0, ttft_ns_sum: 0, running: 0, queued: 0,
    process_bytes: 2_147_483_648,
    process_memory_byte_seconds_total: 2_147_483_648 * (i === 5 ? baseMemorySeconds - 5 : i * 6 * 3600),
    process_memory_observed_seconds_total: i === 5 ? baseMemorySeconds - 5 : i * 6 * 3600,
  }));
  const history = Array.from({ length: 61 }, (_, i) => ({
    at_ms: start + i * 5000,
    continuity_id: 1,
    generation_tokens_live: Math.min(i, 48) * 10 + Math.max(Math.min(i, 56) - 48, 0) * 100,
    prefill_tokens_forwarded_live_total: Math.min(i, 48) * 50 + Math.max(Math.min(i, 56) - 48, 0) * 500,
    decode_active_ns_total: Math.min(i, 56) * 1_000_000_000,
    prefill_active_ns_total: Math.min(i, 56) * 1_000_000_000,
    cache_queries_total: 1000 + Math.min(i, 48) * 10 + (i >= 55 ? 1 : 0),
    cache_hits_total: 900 + Math.min(i, 48),
    ttft_count: Math.min(i, 48) * 2 + (i >= 55 ? 1 : 0),
    ttft_ns_sum: Math.min(i, 48) * 200_000_000 + (i >= 55 ? Math.round(ttftMs * 1_000_000) : 0),
    requests_completed_total: i < 55 ? 0 : 1,
    running: i > 1 && i < 10 ? 1 : 0,
    queued: 0,
    process_bytes: memoryGap && i === 55 ? null : i <= 48 ? 2_147_483_648 : 4_294_967_296,
  }));
  let memorySeconds = baseMemorySeconds, memoryArea = 2_147_483_648 * baseMemorySeconds;
  history.forEach((sample, i) => {
    if (i && sample.process_bytes != null && history[i - 1].process_bytes != null) {
      memorySeconds += 5;
      memoryArea += (sample.process_bytes + history[i - 1].process_bytes) / 2 * 5;
    }
    sample.process_memory_byte_seconds_total = memoryArea;
    sample.process_memory_observed_seconds_total = memorySeconds;
  });
  return {
    counters: { generation_tokens_total: 1280, prefill_tokens_total: 6400, requests_success_total: 1, prefix_cache_queries_total: 1481, prefix_cache_hits_total: 948 },
    gauges: { generation_tokens_live: 1280, prefill_tokens_live: 0, requests_prefilling: 0, requests_running: 0, requests_waiting: 0 },
    histograms: { prefill_time_seconds: { sum: 20 } },
    monitor: {
      server: { sampled_at_ms: now, started_at_ms: started, sample_interval_ms: 2000, instance_id: 'fixture', version: 'fixture', uptime_seconds: 30 * 3600 },
      history_archive: archived,
      history,
      lifetime_totals: history.at(-1),
      recent_requests: [
        { id: 0, model: '<fixture-model>', outcome: 'success', finished_at_ms: started + 4 * 3_600_000, ttft_ms: 500, e2e_ms: 600, queue_ms: 10, output_tokens: 10 },
        { id: 1, model: '<fixture-model>', outcome: 'success', finished_at_ms: start + 275_000, ttft_ms: ttftMs, e2e_ms: ttftMs + 100, queue_ms: 10, output_tokens: 50, prefill_ms: 55, decode_ms: 55 },
      ],
      active_requests: [], models: [], events: [], resources: {}, diagnostics: {}, cache: { queries: 1481, hits: 948 },
      retention: { history_seconds: 3600, archive_compacted: true, archive_sample_interval_ms: 21_600_000, request_capacity: 256, event_capacity: 128, requests_dropped: 300, events_dropped: 20 },
    },
  };
}

const server = createServer((request, response) => {
  if (request.url === '/metrics.json') {
    polls++;
    response.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    response.end(JSON.stringify(snapshot()));
  } else if (request.url === '/v1/models') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ data: [] }));
  } else if (request.url === '/props') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{}');
  } else if (request.url === '/_fixture') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ polls }));
  } else if (request.url === '/' || request.url?.startsWith('/?')) {
    response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    response.end(page);
  } else {
    response.writeHead(404);
    response.end();
  }
});

server.listen(Number(process.env.MONITOR_FIXTURE_PORT || 11267), '127.0.0.1', () => {
  console.log(`Monitor fixture: http://127.0.0.1:${server.address().port}/#monitor`);
});
