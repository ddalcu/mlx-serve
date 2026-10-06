/* The base of every API path the console asks for.

 * A reverse proxy can mount the server below its own root and STRIP that prefix
 * on the way in (`tailscale serve --set-path /mlx http://127.0.0.1:8003`), so
 * the server never learns it: only the URL the page was served under still
 * carries it. A root-absolute request then resolves against the PROXY's origin
 * root, gets the proxy's 404, and the catch reads it as an empty server.
 *
 * ONE definition for the whole page. It lives in the boot script slot because
 * the metrics panel's script runs in the body, before `app.js` exists.
 *
 * index.html is a std.fmt format string, so this is a separate file injected as
 * a runtime `{s}` arg like app.css/app.js. */

// The mount itself is returned without a trailing slash, '' at the origin root.
// A last segment holding a dot is a file, so a file's directory is the mount;
// anything else is the mount. A value that is not a pathname at all (an
// absolute URL, a stubbed `location`) invents no prefix.
function apiPrefix(pathname) {
  var p = String(pathname || '');
  if (p.charAt(0) !== '/') return '';
  while (p.length > 1 && p.charAt(p.length - 1) === '/') p = p.slice(0, -1);
  var slash = p.lastIndexOf('/');
  if (p.indexOf('.', slash) >= 0) p = p.slice(0, slash);
  return p === '/' ? '' : p;
}

// The test harness evaluates each console script inside `new Function`, where a
// declaration is function-scoped; this is what makes it visible to the others.
globalThis.apiPrefix = apiPrefix;
