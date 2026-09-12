/**
 * When the TUI polls. Lifted out of `tui.tsx` so the rules are testable: the
 * component supplies the clock and the flags, these functions decide.
 */

/** Anything still moving: our own turn, or the server working for someone else. */
export function isBusy(hasActive: boolean, now: number, serverBusyUntil: number): boolean {
  return hasActive || now < serverBusyUntil
}

/** Does anything visible need these numbers? */
export function isWanted(instanceIsOurs: boolean, panelOnScreen: boolean, busy: boolean): boolean {
  if (!instanceIsOurs) return false
  return panelOnScreen || busy
}

/** /metrics.json interval: the fast rate while work is in flight, the idle one otherwise. */
export function metricsEveryMs(busy: boolean, pollHz: number, idlePollHz: number): number {
  return 1000 / (busy ? pollHz : idlePollHz)
}

/** A periodic read is due when its interval has passed since the last one. */
export function isDue(now: number, lastAt: number, everyMs: number): boolean {
  return now - lastAt >= everyMs
}

/**
 * Whether a sample showing work should hold the fast poll rate open. Server-wide
 * gauges count another client's work too, so with nothing of ours on screen and
 * nothing of ours running there is nobody to poll for.
 */
export function holdsServerBusy(panelOnScreen: boolean, hasActive: boolean): boolean {
  return panelOnScreen || hasActive
}
