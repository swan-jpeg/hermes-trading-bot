# Impact Agent — Skill + LLM guide

The Impact Agent is a **between-layer** that interprets events and decides
which instruments (stocks, bonds, ETFs) are affected — and with what
sentiment. It is the bridge between a real-world event and the RL model.

> **"Trump says Jensen Huang is a good guy, you can trust him"**
> → the agent understands: this is **positive for NVIDIA (NVDA)**, sentiment +0.8.

## What's in here

```
impact_agent/
├── skill/
│   └── impact-agent-skill.md   ← the SKILL: put this on your own LLM
└── README.md                   ← this guide: how to wire it up
```

## The skill

`skill/impact-agent-skill.md` is a **system prompt** you put on your own LLM
(Qwen3, Llama, GPT, or any capable model — OpenAI-compatible or via a local
server). It gives the LLM the role of an impact analyst with:

- **Output format**: one JSON vector per instrument
  `{entity, asset_class, sentiment, confidence, reason}`
- **Rules**: understand context (not just keywords), sentiment in -1..+1,
  only actually-affected instruments, `[]` when there is no impact.
- **Examples**: Trump→NVDA, government spending→AGG/XLI, no impact→`[]`.

## How to implement it (bring your own LLM / API key)

There is **no bundled model**. You supply the LLM and its API key yourself —
any OpenAI-compatible chat-completions endpoint works, whether that is a
cloud API or a locally hosted model (e.g. Ollama). The bot just needs three
values, which you put in `.env` (secrets only — never in GitHub):

```env
# Your own LLM as the impact agent
IMPACT_LLM_API_KEY=<your-key>
IMPACT_LLM_MODEL=<model-id, e.g. qwen/qwen3-27b or your-local-model>
IMPACT_LLM_BASE_URL=<chat-completions endpoint, e.g. http://localhost:11434/v1>
```

### Step 1 — Put the skill on your LLM
Copy the contents of `skill/impact-agent-skill.md` as the system prompt in
your LLM (or install it as a skill in your agent framework). Your LLM now
acts as the impact agent.

### Step 2 — Wire your LLM into the bot
The bot sends every event to the LLM's chat-completions endpoint and gets the
affected instruments back. Enable it in `config/config.yaml`:

```yaml
impact_use_llm: true
```

The bot reads `IMPACT_LLM_API_KEY`, `IMPACT_LLM_MODEL` and
`IMPACT_LLM_BASE_URL` from the environment (`.env`). If `IMPACT_LLM_BASE_URL`
is empty it defaults to the public chat-completions endpoint
(`https://inference-api.nousresearch.com/v1`).

> You can also pass the model/openai-style values directly in code:
> ```python
> from hermes_bot.impact import LLMImpactAgent
> agent = LLMImpactAgent(model="your-model", base_url="http://localhost:11434/v1")
> ```

### Step 3 — Test
```bash
uv run python -c "
from hermes_bot.impact import LLMImpactAgent
agent = LLMImpactAgent()
r = agent.analyze('Trump says Jensen Huang is a good guy, you can trust him',
                  source='speech', entity_id='trump')
print([i['entity'] for i in r.impacted])
"
```

### Fallback (no LLM needed)
Without an API key the impact agent falls back to **keyword matching** (in
`hermes_bot/impact.py`) — fast, free, offline, and always works.

## The output vector

Every affected entity gets a vector the RL model uses:
```json
{
  "entity": "NVDA",
  "asset_class": "stock",
  "sentiment": 0.8,
  "confidence": 0.7,
  "reason": "Trump praises Jensen Huang — positive for AI chips"
}
```

The RL model sees per stock the sentiment, confidence and quality, and decides
whether to buy, hold or sell.
