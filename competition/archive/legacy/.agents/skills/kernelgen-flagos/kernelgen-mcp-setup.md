<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# KernelGen MCP configuration and runtime verification

This guide separates local configuration from a live KernelGen connection. A config file, successful HTTP handshake, or server-side `tools/list` response does not prove that the current agent can call the tool. Code generation may start only after the required operation is visible in the active tool registry.

## Choose the client configuration

### Codex

Codex reads MCP servers from the user-level `config.toml`; it does not load this checkout's `.mcp.json`. Merge this block into `~/.codex/config.toml` without replacing other entries:

```toml
[mcp_servers.kernelgen-server]
url = "https://kernelgen.flagos.io/sse/"
enabled = true
bearer_token_env_var = "KERNELGEN_TOKEN"
startup_timeout_sec = 30
tool_timeout_sec = 600
```

Set `KERNELGEN_TOKEN` in the environment that launches Codex. If the desktop app does not inherit that environment, replace `bearer_token_env_var` with a `http_headers_helper` command that reads the token from a local private credential source and prints only a JSON object such as `{"Authorization":"Bearer …"}` to stdout. Keep the helper outside Git and restrict its file permissions. Do not configure both auth methods; an explicit bearer token takes precedence over helper-provided Authorization.

After changing the user config, reload Codex or start one new local task. Verify the server is enabled with `codex mcp list`, then confirm the needed operation is present in the new task's live tool registry. The command-line server list is a configuration/connection check; it does not replace the live tool check.

Official references: [Codex MCP setup](https://developers.openai.com/learn/docs-mcp) and [Codex config reference](https://learn.chatgpt.com/docs/config-file/config-reference).

### Clients that read the project `.mcp.json`

`competition/task78/kernelgen/mcp.json.example` is for clients that support this project-level JSON format. It uses environment-variable expansion so the token stays out of the file. For Claude Code, set `KERNELGEN_TOKEN` before launching the client and approve the project MCP server when prompted. See the [Claude Code MCP guide](https://docs.anthropic.com/en/docs/claude-code/mcp).

Other clients may use a different file, scope, or transport name. Follow that client's current documentation; do not copy a Codex TOML block into a JSON config or assume a project `.mcp.json` is shared across applications.

## Verify the runtime

1. Check the configuration location for the active client. Never print or paste the token while inspecting it.
2. Check that the client reports `kernelgen-server` enabled/connected. For Codex, `codex mcp list` checks the user-level server entry.
3. In the active agent task, confirm the exact required operation is callable: `generate_kernel`, `optimize_kernel`, `specialize_kernel`, or `autotune_kernel`. Use the operation needed by the selected workflow.
4. Record the result as `absent`, `configured_unavailable`, or `callable` in the run record. Only `callable` permits KernelGen generation.

A successful direct HTTP `initialize` or `tools/list` request is useful service-side evidence, but it does not prove that the current agent loaded the server. If the operation is missing, stop generation and keep only read-only diagnosis and experiment planning.

## When the server is configured but the tool is missing

- If the config was just added or changed, reload the client once and check a new task's live tool registry.
- If it is still missing, inspect that client's MCP connection/startup diagnostics and the exact transport/auth error. Do not repeatedly restart, request a new token, or rewrite a working config without evidence of a specific fault.
- If the endpoint is reachable and lists the expected operations but the active task still lacks them, report `configured_unavailable`. Preserve the candidate source and do not switch to hand-authored code unless the user explicitly authorizes that method for the run.
- Never claim generation from configuration, a network probe, or a tool name mentioned in documentation. Save the raw response and returned source only after an actual registered tool call.

## Credential handling

Never ask the user to paste a KernelGen token into chat. Keep real credentials in a local environment, ignored file, OS credential store, or private helper. Do not commit `.mcp.json`, Codex config, helper scripts containing credentials, request headers, or token-bearing logs. The tracked example contains only an environment-variable reference.
