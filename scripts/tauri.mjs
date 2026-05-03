import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const cargo = await cargoCommand();

const exitCode = await run(cargo, ["tauri", ...args]);
if (args[0] === "build") {
  const cleanupExitCode = await run(process.execPath, [join(root, "scripts", "clean-tauri-target.mjs")]);
  if (cleanupExitCode !== 0) {
    process.exit(cleanupExitCode);
  }
}

process.exit(exitCode);

function run(command, commandArgs) {
  return new Promise((resolve) => {
    const child = spawn(command, commandArgs, {
      cwd: root,
      env: process.env,
      stdio: "inherit",
    });

    child.on("error", (error) => {
      console.error(error.message);
      resolve(1);
    });
    child.on("close", (code) => resolve(code ?? 1));
    });
}

async function cargoCommand() {
  const bundledCargo = "/Users/macstudio/.cargo/bin/cargo";
  try {
    await access(bundledCargo);
    return bundledCargo;
  } catch {
    return "cargo";
  }
}
