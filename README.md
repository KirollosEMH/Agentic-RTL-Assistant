# Agentic RTL Assistant

Agentic RTL Assistant is a local, configuration-driven engineering application for understanding
and generating Verilog RTL. Its primary architecture is **Multi-Agent + LangGraph + GraphRAG**:
focused LLM agents handle classification, explanation, generation, and repair, while deterministic
Python services own source access, PyVerilog parsing, graph construction, retrieval, validation, and
telemetry.

The starter RTL in `descriptive_verilog_design/` remains the authoritative source project. The
knowledge graph is a lazy, derived index and is rebuilt when source hashes change.

## Setup

Python 3.12 is required.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
uv sync
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install uv
uv sync
```

Once `uv` is available, manually activating the environment is optional when using `uv run`.

## Configuration

`config/default.yaml` is complete and selects `multi_agent_graphrag`. Its model profiles use
`gpt-oss:120b-cloud` through the signed-in local Ollama gateway. Run `ollama signin` and
`ollama pull gpt-oss:120b-cloud` before using the default configuration. Use `config/local.yaml`
to select different provider/model profiles.

Configuration precedence currently implemented is default YAML, selected YAML, then these
environment overrides:

- `RTL_ASSISTANT_PROJECT_ROOT`
- `RTL_ASSISTANT_APPROACH`
- `RTL_ASSISTANT_LOG_LEVEL`

Provider credentials are named in `.env.example` but are never stored in YAML. Export them in the
shell (or load a local `.env` by your own environment tooling).

The files in `config/experiments/` override only the architecture family:

- A0 `direct_llm` (one call with every project RTL file in the prompt)
- A1 `text_rag`
- A2 `single_agent` (one agent with bounded read-only source tools)
- A3 `multi_agent_rag`
- A4 `multi_agent_graphrag` (primary)

## Running

Launch the Textual TUI:

```bash
uv run rtl-assistant --config config/default.yaml
```

At startup, paste a project directory into the path field or select one from the directory tree,
then choose **Open project**. The configured YAML project root is only the initial selection in
interactive mode.

The TUI keeps a bounded in-memory conversation session for follow-up questions. Use **New
session** to clear the active conversation context. Opening another project also starts a fresh
session. The number of retained user/assistant messages is controlled by
`context.max_conversation_messages`.

Run one headless request:

```bash
uv run rtl-assistant --config config/default.yaml --ask "What modules are implemented in the project?"
```

Run the required evaluation dataset and persist resolved configuration, metrics, traces, and
results under `runs/`:

```bash
uv run rtl-assistant eval --config config/experiments/multi_agent_graphrag.yaml
```

Run the full approach/model matrix across the configured Ollama and OpenRouter models:

```bash
uv run --env-file .env rtl-assistant eval --config config/eval-matrix.yaml
```

The runner evaluates the cross product of `evaluation.approaches` and
`evaluation.model_profiles`. Each named model profile represents one provider/model pair and is
assigned to every agent role for that matrix cell, making comparisons consistent. Add another
model by defining another entry under `models.profiles` and including its name in
`evaluation.model_profiles`.

The matrix creates a parent run under `runs/matrix/` with aggregate `results.json`,
`metrics.json`, and `traces.jsonl`. The `combinations/` directory contains separate resolved
configuration and artifacts for every approach/model pairing. A failed request is recorded without
aborting the remaining combinations.

Evaluation cases can label `expected_source_paths`, `expected_entities`, and
`expected_relations`. When `retrieval` is enabled in `evaluation.metrics`, `results.json` contains
per-case source precision/recall/F1, hit rate, MRR, entity recall, relation recall, and a macro
retrieval-accuracy score. `metrics.json` contains the corresponding averages across evaluated
requests.

## Architecture

The YAML loader produces a validated `AppConfig`; factories then inject model providers and
deterministic services into the selected approach. The A4 workflow is bounded by configured graph
steps and repair attempts. It classifies intent, refreshes the derived graph if source hashes have
changed, resolves entities, traverses a bounded neighborhood, maps graph entities back to exact
source ranges, and gives the compact evidence pack to one specialized agent. Generated RTL is
parser-validated outside the code agent before an optional repair pass.

The model factory normalizes OpenAI, Ollama, OpenRouter, Groq, and Cloudflare responses into common
request/response and token-usage types. OpenAI uses the Responses API; the other configured
remote/local adapters use their OpenAI-compatible endpoints. Cached input tokens remain `null`
when a provider does not report them. Credential-free deterministic model doubles exist only under
`tests/` and are injected through the same provider boundary.

Telemetry events are separate from logs and include request, agent, retrieval, and validation
activity. The TUI consumes these events, while the evaluation runner consumes the common
`RunResult`; neither depends on the other.

## Current vertical slice

With Ollama signed in and the configured cloud model available, the application can discover and
parse the three starter modules, answer the required module-list/hierarchy/control questions using
graph-and-source evidence, generate a `FifoBuffer`, validate its syntax/module name, display
activity and token usage, and run the required evaluation cases. Reads are confined to the
configured project root and generated RTL is displayed only; it is not written automatically.

The A2 tool loop uses a provider-independent JSON protocol so it works through OpenAI, Ollama,
OpenRouter, Groq, and Cloudflare without provider-specific tool-call payloads. This first phase
intentionally leaves embedding/vector retrieval, persistent graph backends, compiler/simulator
integration, safe confirmed writes, rich conversation summarization, provider-specific advanced
options, and statistical/behavioral evaluators for later work.

## Quality checks

```bash
uv run pytest
uv run ruff check .
```

The supplied `parse_with_pyverilog.py` utility uses PyVerilog's external preprocessing path and may
require `iverilog`. The application parser adapter parses source text directly for the current
structural slice, so the normal demo and tests do not require that executable.
