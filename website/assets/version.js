// The nav version pill resolves itself from the latest GitHub release, the same
// source the Download CTA points at, so no page carries a version string that
// can go stale between releases.
//
// It fails silent on purpose: an unresolved pill is empty, and `.nav-ver:empty`
// hides it. Showing nothing beats showing a wrong version.
(function () {
  var pills = document.querySelectorAll('.nav-ver');
  if (!pills.length) return;

  var KEY = 'mlxserve-latest-release';
  var TTL = 6 * 60 * 60 * 1000; // GitHub allows 60 unauthenticated calls/hour

  function paint(v) {
    pills.forEach(function (p) { p.textContent = v; });
  }

  try {
    var cached = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (cached && cached.v && Date.now() - cached.at < TTL) {
      paint(cached.v);
      return;
    }
  } catch (e) { /* private mode / bad JSON — just fetch */ }

  fetch('https://api.github.com/repos/ddalcu/mlx-serve/releases/latest')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (d) {
      var v = d && d.tag_name;
      if (!v) return;
      paint(v);
      try { localStorage.setItem(KEY, JSON.stringify({ v: v, at: Date.now() })); } catch (e) {}
    })
    .catch(function () {});
})();
