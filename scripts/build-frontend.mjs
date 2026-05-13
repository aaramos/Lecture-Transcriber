import { mkdir, cp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist");

await rm(dist, { force: true, recursive: true });
await mkdir(join(dist, "src"), { recursive: true });
await cp(join(root, "index.html"), join(dist, "index.html"));
await cp(join(root, "src", "main.js"), join(dist, "src", "main.js"));
await cp(join(root, "src", "token_metrics.js"), join(dist, "src", "token_metrics.js"));
await cp(join(root, "src", "styles.css"), join(dist, "src", "styles.css"));

console.log("Built frontend to dist/");
