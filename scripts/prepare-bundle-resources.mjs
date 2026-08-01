import { execFile } from "node:child_process";
import { copyFile, mkdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const stagingRoot = resolve(root, "src-tauri", "bundle-resources");
const projectRoot = join(stagingRoot, "project");

if (projectRoot === root || !projectRoot.startsWith(`${stagingRoot}/`)) {
  throw new Error(`Refusing to replace unsafe bundle staging path: ${projectRoot}`);
}

await rm(projectRoot, { recursive: true, force: true });

const { stdout } = await execFileAsync("git", ["ls-files", "src/lecture_processor/**"], {
  cwd: root,
});
const trackedProcessorFiles = stdout.split("\n").filter(Boolean);
const requiredFiles = [
  "README.md",
  "pyproject.toml",
  ".tools/darwin_arm64/ffmpeg",
  ".tools/darwin_arm64/ffprobe",
  ...trackedProcessorFiles,
];

for (const relativePath of requiredFiles) {
  const source = join(root, relativePath);
  const destination = join(projectRoot, relativePath);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

console.log(`Prepared ${requiredFiles.length} tracked release resources.`);
