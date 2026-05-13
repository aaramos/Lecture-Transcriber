import assert from "node:assert/strict";

import { formatTokenCount, tokenMetricLabel } from "../src/token_metrics.js";

assert.equal(formatTokenCount(25_400), "25K");
assert.equal(formatTokenCount(10_400), "10K");
assert.equal(formatTokenCount(8_840), "8.8K");

assert.equal(tokenMetricLabel({ sent: 891, received: 80 }, 0.002, { compact: true }), "in 891 · out 80");
assert.equal(tokenMetricLabel({ sent: 8_800, received: 720 }, 0.9, { compact: true }), "in 8.8K · out 720");
assert.equal(
  tokenMetricLabel({ sent: 8_800, received: 720 }, 16.8, { compact: true }),
  "in 8.8K · out 720 · 42.9 out tok/s",
);
assert.equal(tokenMetricLabel({ sent: 8_800, received: 0 }, 16.8, { compact: true }), "in 8.8K · out 0");
assert.equal(tokenMetricLabel({ sent: 0, received: 0 }, 16.8, { compact: true }), "");

console.log("Token metric tests passed.");
