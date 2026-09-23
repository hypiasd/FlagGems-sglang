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

# KernelGen MCP Configuration Check & Auto-Setup

This file checks project-local configuration and explains setup. A config file
only proves that configuration was written; it does not prove that the current
agent session connected the server or exposed its tools. Runtime availability
must be checked in the current tool registry before calling KernelGen.

---

## Step 1: Check Whether MCP Is Already Configured

Use the Read tool to check the following files in order (only check project-local paths — do not read the user's home directory):

1. `.mcp.json`
2. `.claude/settings.json`

For each file:
- If the file does not exist, skip it
- If the file exists, parse the JSON and check whether `mcpServers` contains a key that includes `kernelgen` (case-insensitive)

**Decision rules**:
- Found in any file → record **configured** and continue to the runtime check below.
- Not found in either file → **not configured**, proceed to Step 2.

After reading configuration, check whether the required operation (`generate_kernel`,
`optimize_kernel`, or `specialize_kernel`) is actually visible and callable in the
current agent's tool registry. Never infer this from the JSON key alone.

- Configured and callable → continue to the selected sub-skill.
- Configured but not callable → report **configured, runtime unavailable**. Do not
  ask for or rotate the token, rewrite the config, or claim that a restart will
  definitely fix it. If this config was just added or changed, ask the user to
  restart/reload once; otherwise check the client connection and resume after the
  tool is exposed. Code generation must remain stopped, though read-only diagnosis
  and an experiment plan may continue.
- Not configured → proceed to Step 2.

---

## Step 2: Guide the User to Configure a Local Token

Never ask the user to paste a KernelGen Token into chat and never write a real
token into a tracked file. Tell the user to copy
`competition/task78/kernelgen/mcp.json.example` to the project-root
`.mcp.json`, replace the placeholder locally, and keep `.mcp.json` untracked.

Output the following message when configuration is absent:

```
The KernelGen MCP toolset is not yet configured for this checkout.

Create a local, untracked project-root .mcp.json from
competition/task78/kernelgen/mcp.json.example and replace the token locally.
Do not paste the token into chat or commit .mcp.json. Then restart the agent.
```

Stop code generation here. Read-only diagnosis may continue, but do not claim a
KernelGen run until the current agent can call the required tool.

---

## Step 3: Configuration Shape

The local configuration must use the following shape, with the real token
filled in locally:

**Target configuration format** (written to `.mcp.json`):

```json
{
  "mcpServers": {
    "kernelgen-server": {
      "type": "sse",
      "url": "https://kernelgen.flagos.io/sse/",
      "headers": {
        "Authorization": "Bearer <USER_TOKEN>"
      }
    }
  }
}
```

**Important notes**:
- The MCP service URL is fixed as `https://kernelgen.flagos.io/sse/` — the user does not need to provide it
- The example server key is `kernelgen-server`; the actual operation names must still be taken from the live tool registry, not inferred from this key.
- Never overwrite other configuration entries in the file

---

## Step 4: Prompt the User to Restart

After the user has configured the local file, output the following to the user:

```
MCP configuration should now be present in the local .mcp.json. Please restart the agent for the configuration to take effect, then re-run the command.
```

**Stop code generation here.** When the user restarts and re-triggers the skill,
check the live tool registry again; do not assume that configuration visibility
means runtime availability.
