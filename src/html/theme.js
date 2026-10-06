(() => {
  const {readPreferences} = studioModules["src/ui/state.js"];
  const prefs = readPreferences({getItem(key) { try { return localStorage.getItem(key); } catch { return null; } }});
  document.documentElement.dataset.theme = prefs.theme === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : prefs.theme;
  document.documentElement.dataset.accent = prefs.accent;
})();
