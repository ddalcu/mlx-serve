/* Theme boot for the built-in console.
 *
 * Lives in the <head>, before the body paints, so a stored choice never flashes
 * the other theme. Injected as a runtime `{s}` arg like app.css/app.js:
 * index.html is a std.fmt format string, so an inline script would have to
 * double every brace it contains. */

(function () {
  var KEY = 'mlx-serve-theme'
  var root = document.documentElement
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null

  function stored() {
    try {
      var v = localStorage.getItem(KEY)
      return v === 'light' || v === 'dark' ? v : null
    } catch (e) {
      return null
    }
  }

  function current() {
    return stored() || (media && media.matches ? 'light' : 'dark')
  }

  function apply(theme) {
    root.setAttribute('data-theme', theme)
  }

  apply(current())

  // With nothing stored the OS decides, including while this page is open.
  if (media && media.addEventListener) {
    media.addEventListener('change', function () {
      if (!stored()) apply(current())
    })
  }

  window.mlxTheme = {
    current: current,
    toggle: function () {
      var next = current() === 'light' ? 'dark' : 'light'
      try {
        localStorage.setItem(KEY, next)
      } catch (e) {}
      apply(next)
      return next
    }
  }
})()
