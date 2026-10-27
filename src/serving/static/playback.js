/** Presentation only: never change canonical event/evidence clocks. */
export const DEFAULT_EVENT_PREROLL_SECONDS = 2;
export function getPlaybackAnchorTime(timestamp, event, fps) {
  const canonical = Number.isFinite(timestamp) && timestamp >= 0 ? timestamp : 0;
  if (!event || !Number.isFinite(fps) || fps <= 0) return canonical;
  const frame = event.source === 'transfer' || event.source === 'trajectory'
    ? event.metrics?.start_frame : event.source === 'tactical_state_engine'
      ? event.tactical?.evidence_start_frame : null;
  // Formation fitting windows describe historical sampling, not transition onset.
  const published = Number.isFinite(event.timestamp) ? event.timestamp : canonical;
  return Number.isInteger(frame) && frame >= 0 && frame / fps <= published
    ? frame / fps : canonical;
}
export function getPlaybackSeekTime(timestamp, event, fps, duration, preRoll = DEFAULT_EVENT_PREROLL_SECONDS) {
  const roll = Number.isFinite(preRoll) ? Math.max(0, Math.min(10, preRoll)) : DEFAULT_EVENT_PREROLL_SECONDS;
  const end = Number.isFinite(duration) ? Math.max(0, duration) : Infinity;
  return Math.min(end, Math.max(0, getPlaybackAnchorTime(timestamp, event, fps) - roll));
}
