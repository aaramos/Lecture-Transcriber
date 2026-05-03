import { access, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const javascriptFiles = [
  "src/main.js",
  "scripts/build-frontend.mjs",
  "scripts/clean-tauri-target.mjs",
  "scripts/dev-server.mjs",
  "scripts/lint.mjs",
  "scripts/tauri.mjs",
];

const jsonFiles = [
  "package.json",
  "package-lock.json",
  "src-tauri/tauri.conf.json",
];

for (const file of javascriptFiles) {
  await checkExists(file);
  await checkJavaScript(file);
}

for (const file of jsonFiles) {
  await checkExists(file);
  await checkJson(file);
}

console.log(`Lint passed: ${javascriptFiles.length} JavaScript files and ${jsonFiles.length} JSON files checked.`);

async function checkExists(file) {
  await access(join(root, file));
}

async function checkJavaScript(file) {
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--check", join(root, file)], {
      cwd: root,
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${file} failed JavaScript syntax check.`));
    });
  });
}

async function checkJson(file) {
  try {
    JSON.parse(await readFile(join(root, file), "utf8"));
  } catch (error) {
    throw new Error(`${file} contains invalid JSON: ${error.message}`);
  }
}
