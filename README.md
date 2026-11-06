# Pareto Inference for Hermes Agent

Adds **Pareto Inference** to the Hermes provider list: GLM 5.3 Flash, DeepSeek V4.1 Flash and
MiMo V2.6 Flash, pay per token ([pricing](https://docs.paretoinference.com/pricing)).

## Install

```bash
hermes plugins install pareto
echo 'PARETO_API_KEY=your_key' >> ~/.hermes/.env
hermes model                                        # pick "Pareto Inference"
```

Or set it directly in `~/.hermes/config.yaml`:

```yaml
model:
  provider: pareto
  default: z-ai/glm-5.3-flash
auxiliary:
  # Pareto serves streamed requests only. Hermes' auxiliary tasks (titles, compression,
  # vision) are non-streamed by default, so list the endpoint here.
  stream_only_base_urls:
    - api.paretoinference.com
```

## What it does

- Registers one model provider, `pareto` (aliases `pareto-inference`, `paretoinference`), at
  `https://api.paretoinference.com/v1` with `PARETO_API_KEY`.
- Maps `/reasoning` levels to Pareto's top-level `reasoning_effort` (`none` … `max`; `ultra`
  is sent as `max`). With no level set, Pareto's per-model default applies.
- Accepts bare model names (`glm-5.3-flash`) and sends the `vendor/model` id the API requires.

## Disclosure

Hermes sends prompts, tool results and `/models` requests to `api.paretoinference.com` with your
`PARETO_API_KEY`, which Hermes reads. The plugin opens no connections, reads or writes no files,
starts no processes and adds no telemetry.
