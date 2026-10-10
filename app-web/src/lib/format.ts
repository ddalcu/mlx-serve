import { t } from "./i18n/i18n";

export const fmt = (n: number | null | undefined) => (n == null || !Number.isFinite(n) ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 2 }));

export const bytes = (n: number | null | undefined) =>
  n == null ? "—" : n >= 1073741824 ? t("%@ GiB", [fmt(n / 1073741824)]) : t("%@ MiB", [fmt(n / 1048576)]);

/** How alarming a percentage is: the meters color themselves by it. */
export const level = (n: number | null | undefined) => (n != null && n >= 90 ? "critical" : n != null && n >= 70 ? "warning" : "normal");
