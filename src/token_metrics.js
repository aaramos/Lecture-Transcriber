const MIN_TOKEN_RATE_SECONDS = 1;

export function tokenMetricLabel(usage, seconds, options = {}) {
  const sent = Number(usage?.sent || 0);
  const received = Number(usage?.received || 0);
  const tokens = sent + received;
  if (tokens <= 0) return "";

  const parts = [
    `in ${formatTokenCount(sent)}`,
    `out ${formatTokenCount(received)}`,
  ];
  const elapsed = Number(seconds || 0);
  if (received > 0 && Number.isFinite(elapsed) && elapsed >= MIN_TOKEN_RATE_SECONDS) {
    parts.push(`${(received / elapsed).toFixed(1)} out tok/s`);
  }
  return options.compact ? parts.join(" · ") : parts.join(" · ");
}

export function formatTokenCount(value) {
  const count = Math.max(0, Math.round(Number(value) || 0));
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 10_000) return `${Math.round(count / 1_000)}K`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
}
