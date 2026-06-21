# cursor-claude-adapter

Use your own Anthropic-format relay as a model provider in Cursor (BYOK).

## How it works

```
Cursor (OpenAI Chat Completions)  --[Override OpenAI Base URL]-->  this adapter
  --> converts to Anthropic Messages -->  your upstream relay (/v1/messages)
  --> converts the Anthropic response (stream or not) back to OpenAI -->  Cursor
```

## Quick start

```bash
cp .env.example .env   # edit if needed
docker compose up -d --build
```

Expose it over HTTPS (Cursor needs a public URL):

```bash
cloudflared tunnel --url http://localhost:3000
```

## Cursor setup

Settings -> Models:

- **Override OpenAI Base URL**: `https://your-domain/v1`
- **OpenAI API Key**: the key for your upstream relay (passed through as-is)
- Pick a model from the list
- Turn off the Anthropic API Key field

## Models

`/v1/models` is generated dynamically: every base model is exposed with a "no thinking"
variant plus one variant per thinking level. To add a model, add one line to `baseModels`
in `main.go` — no config needed.

### Thinking levels

Add a suffix to the model name to turn on adaptive thinking at that level:
`-low`, `-medium`, `-high`, `-xhigh`, `-max`. No suffix means no thinking.
(`xhigh` is the Claude Code default and works best for coding.)

```
cursor-claude-opus-4-8         -> claude-opus-4-8, no thinking
cursor-claude-opus-4-8-xhigh   -> claude-opus-4-8 + adaptive thinking, effort xhigh
```

## Auth

The key is passed through. Whatever you put in Cursor's OpenAI API Key field is forwarded
upstream as `x-api-key`. The adapter never stores it.

## Configuration

All variables have defaults; set them in `.env` / `docker-compose` for production.

| Variable | Default | Description |
| --- | --- | --- |
| `UPSTREAM_URL` | `https://api.anthropic.com` | Upstream host only. The `/v1/messages` path is added in code — do not append it. |
| `MODEL_PREFIX` | `cursor-` | Prefix stripped before forwarding (`cursor-claude-opus-4-8` -> `claude-opus-4-8`). |
| `ANTHROPIC_VERSION` | `2023-06-01` | `anthropic-version` header. |
| `PORT` | `3000` | Listen port. The container is always 3000; the published port is set in compose. |
| `DEBUG` | `0` | Set to `1` to log request/response summaries. |

## Notes

No third-party dependencies (`main.go` for entry/convert/forward, `util.go` for pure
helpers). `go.mod` only exists because Go needs it for the module declaration.
