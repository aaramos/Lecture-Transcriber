import { readdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const target = resolve(root, "src-tauri", "target");
const release = join(target, "release");

const pathsToRemove = [
  join(target, "debug"),
  join(release, "build"),
  join(release, "deps"),
  join(release, "incremental"),
  join(release, "_up_"),
];

for (const path of pathsToRemove) {
  await removeInsideTarget(path);
}

await removeReleaseHashFiles();

console.log("Cleaned Tauri build intermediates; kept release app, DMG, and binary.");

async function removeReleaseHashFiles() {
  let entries = [];
  try {
    entries = await readdir(release, { withFileTypes: true });
  } catch {
    return;
  }

  const removableNames = entries
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) => /\.(d|rmeta|rlib|o)$/.test(name) && name !== "lecture-processor-app.d");

  for (const name of removableNames) {
    await removeInsideTarget(join(release, name));
  }
}

async function removeInsideTarget(path) {
  const resolved = resolve(path);
  if (resolved === target || !resolved.startsWith(`${target}/`)) {
    throw new Error(`Refusing to remove path outside Tauri target: ${path}`);
  }
  await rm(resolved, { recursive: true, force: true });
}
