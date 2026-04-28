# 03 — Credential Storage Design

## 1. Threat Model

The user enters an API key. We must:

- Never write the key to a plaintext config file or to log output.
- Never include the key in CLI args, environment variables visible in `ps`, or in error messages.
- Never transmit the key except in the HTTPS request to the provider.
- Survive the user copying the app's user-data folder to another machine without the key going with it.

We do **not** need to defend against:

- A user with shell access to the same macOS account (they can already read keychain items they own).
- Memory inspection of the running process.
- A compromised provider SDK.

These are out of scope for an opt-in desktop study tool with no shared accounts.

## 2. Storage Backend

| Platform | Backend                            | Library                 |
|----------|------------------------------------|-------------------------|
| macOS    | macOS Keychain (login keychain)    | Apple Security framework via `security-framework` Rust crate |
| Windows  | Windows Credential Manager         | `windows-rs` `Credential*` APIs |
| Linux    | Secret Service (libsecret)         | `secret-service` Rust crate |

Phase 1 ships macOS only. Windows and Linux backends are architected for but not implemented in MVP.

The reason credentials live in Rust, not Python, is that:

- Secrets must never appear in CLI args. Python is launched as a subprocess; passing the key as an arg or env var is a leak vector.
- The Rust process is the long-lived parent; the secret stays in its memory and is sent to Python via stdin (write-once, immediately consumed) or via a short-lived pipe.

## 3. Tauri Commands

Three new Tauri commands live in `src-tauri/src/keychain.rs`:

```rust
#[tauri::command]
fn keychain_set(provider: String, api_key: String) -> Result<(), String>;

#[tauri::command]
fn keychain_get(provider: String) -> Result<Option<String>, String>;
// Returns None when no key is stored; never returns "" to mean missing.

#[tauri::command]
fn keychain_delete(provider: String) -> Result<bool, String>;
// Returns true if a key was deleted, false if none was stored.
```

Service identifier convention: `com.lecture-processor.api-key.<provider>` where `<provider>` is `anthropic`, `gemini`, or `grok`. The user's macOS account is the keychain owner.

The frontend never holds the key in memory longer than necessary. The flow:

1. User pastes key in Settings dialog.
2. JS calls `invoke("keychain_set", { provider, apiKey })`.
3. JS clears the input field and discards the local string.
4. Subsequent enrichment runs the key from keychain to the Python subprocess via stdin (see §5).

## 4. Validation Before Storage

`keychain_set` MUST NOT validate the key against the provider's API. That belongs to a separate `test_provider` Tauri command:

```rust
#[tauri::command]
async fn test_provider(provider: String, model: String) -> Result<TestResult, String>;
```

`test_provider` reads the stored key, spawns the Python CLI with subcommand `provider-test`, and returns the result. Separating storage from validation lets the user save a key now and test connectivity later (e.g., when offline). It also lets us re-test without prompting for the key again.

## 5. Passing The Key To Python

The Python subprocess receives the key on a dedicated stdin write at process start, then closes that write half:

```rust
// In run_enrichment() in lib.rs, after spawning child:
let stdin = child.stdin.take().expect("piped");
let mut writer = BufWriter::new(stdin);
writeln!(writer, "{}", api_key)?;  // single line, single key
writer.flush()?;
drop(writer);                       // close stdin so Python knows to stop reading
```

Python's CLI subcommand `enrich` reads exactly one line from stdin at startup, holds it in a local variable, and immediately closes stdin. The key never appears in `ps`, never in environment, never on the filesystem. It's only in the parent Rust process memory and the child Python process memory.

For the `process` (local-only) subcommand, no key is read — stdin is left as the inherited terminal stdin so existing behavior is unchanged.

A flag `--read-key-from-stdin` makes the contract explicit:

```bash
lecture-processor enrich /path/to/output --provider anthropic --read-key-from-stdin
```

## 6. What We Do NOT Do

- **Do not store the key in `localStorage` or any browser-side persistence.** Tauri webviews can be inspected; treat localStorage as plaintext.
- **Do not write a fallback config file like `~/.lecture-processor/keys.json`.** Falling back from keychain to plaintext defeats the purpose. If keychain is unavailable, the user enters the key per session.
- **Do not log the key, even truncated.** Logs get pasted into bug reports. There is no safe way to log "first 4 chars of the key" — those characters identify the account.
- **Do not pass the key as `--api-key` on the command line.** This is the most common leak in shell tools and the reason we use stdin.

## 7. Multiple Keys / Profiles

Phase 1 supports **one key per provider, per OS account.** A user who has multiple Anthropic keys for different projects re-enters when switching. This is a deliberate simplification — multi-profile UX needs design that's not in scope for MVP.

If multi-profile is requested in Phase 3+, the service identifier changes to `com.lecture-processor.api-key.<provider>.<profile-name>` and the UI gains a profile picker. The keychain storage primitives don't change.

## 8. Removal Behavior

- Settings dialog has a "Remove" button next to each stored key. It calls `keychain_delete`.
- Uninstalling the app does **not** automatically remove keychain items. The user can remove them via Keychain Access or the in-app button. Document this in release notes.
- A future "Reset all settings" action MUST iterate every registered provider and call `keychain_delete` on each.

## 9. Windows Implementation Outline (For Phase 3+)

- Crate: `windows` with the `Win32_Security_Credentials` feature.
- `CredWriteW` to set, `CredReadW` to get, `CredDeleteW` to delete.
- Credential type: `CRED_TYPE_GENERIC`.
- TargetName: same naming convention as macOS service identifier.
- Persist: `CRED_PERSIST_LOCAL_MACHINE` is wrong — use `CRED_PERSIST_ENTERPRISE` so the user's domain credentials roam where appropriate, or `CRED_PERSIST_LOCAL_MACHINE` for non-roaming. Default to `CRED_PERSIST_LOCAL_MACHINE` and document.

The Rust trait abstraction looks like:

```rust
trait SecretStore: Send + Sync {
    fn set(&self, service: &str, secret: &str) -> Result<(), SecretStoreError>;
    fn get(&self, service: &str) -> Result<Option<String>, SecretStoreError>;
    fn delete(&self, service: &str) -> Result<bool, SecretStoreError>;
}
```

Implementations: `MacOsKeychainStore`, `WindowsCredentialStore`, `LinuxSecretServiceStore`. The Tauri commands hold an `Arc<dyn SecretStore>` chosen at startup based on `cfg!(target_os = ...)`. This is exactly the abstraction the architect handoff asks for ("room for Windows later").

## 10. Tests

- Unit tests for the macOS implementation against a temporary keychain (the `security-framework` crate supports creating an in-memory test keychain).
- Round-trip test: set, get, delete, get-returns-None.
- Test that overwriting an existing key works (set twice).
- Test that deleting a nonexistent key returns `Ok(false)`, not an error.
- Test that the key never appears in stdout/stderr of the test process.

For Windows and Linux backends in Phase 3+, the same test suite runs against a per-platform `SecretStore` implementation. The trait makes this possible without conditional test code.
