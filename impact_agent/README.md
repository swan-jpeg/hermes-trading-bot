# Impact Agent — Skill voor je eigen LLM

Dit is een **skill** die je op je eigen LLM (bv. Qwen3-27B, Llama-3.1, of elk
groot model) kunt zetten, zodat die als **Impact Agent** functioneert: een
tussenlaag die gebeurtenissen interpreteert en bepaalt welke instrumenten
(stocks, obligaties, ETF's) geraakt worden — en met welk sentiment.

## Wat de Impact Agent doet

De Impact Agent is de schakel tussen een gebeurtenis en de beïnvloede
instrumenten. Voorbeeld:

> "Trump zegt dat Jensen Huang een goede gozer is en je hem kan vertrouwen"
>
> → De agent begrijpt: dit is **positief voor NVIDIA (NVDA)**, sentiment +0.8.

Zonder deze agent weet het RL-model niet welke stocks geraakt worden door een
speech of rapport. De agent levert die koppeling als een **vector** per
instrument.

## Hoe het werkt

```
Gebeurtenis (speech/rapport/overheidsuitgave/nieuws)
        ↓
IMPACT-AGENT (jouw LLM met deze skill)
        ↓
per beïnvloede entiteit: vector
  {entity, asset_class, sentiment, confidence, reason}
        ↓
RL-model (per entiteit: welk instrument + gewicht + zekerheid)
        ↓
risk engine + Monte Carlo → portfolio → executie
```

## De skill (prompt) — zet dit op je LLM

Kopieer de inhoud van `skill/impact-agent-skill.md` naar je LLM (als
system-prompt, of als skill in je agent-framework). De LLM krijgt dan de rol
van impact-analist en antwoordt in gestructureerd JSON.

## Installatie

### Optie A — Gebruik je eigen LLM als impact agent (aanbevolen)
1. Zet de skill (`skill/impact-agent-skill.md`) op je LLM (Qwen3-27B, etc.).
2. Configureer de bot om je LLM te gebruiken als impact agent:
   - Zet in `.env`:
     ```
     IMPACT_LLM_API_KEY=<jouw key>
     IMPACT_LLM_MODEL=<jouw model, bv. qwen/qwen3-27b>
     IMPACT_LLM_BASE_URL=<jouw endpoint>
     ```
   - Zet in `config/config.yaml`:
     ```yaml
     impact_use_llm: true
     ```
3. De bot stuurt elke gebeurtenis naar je LLM (met de skill als system-prompt)
   en krijgt de beïnvloede instrumenten terug.

### Optie B — Keyword-matching (geen LLM nodig)
`ImpactAgent` in `hermes_bot/impact.py` matcht keywords/sectoren. Snel,
gratis, offline. Werkt altijd als fallback.

## De output-vector

Elke beïnvloede entiteit krijgt een vector:
```json
{
  "entity": "NVDA",
  "asset_class": "stock",
  "sentiment": 0.8,
  "confidence": 0.7,
  "reason": "Trump prijst Jensen Huang (positief voor AI-chips)"
}
```

Dit is de input voor het RL-model: het ziet per stock het sentiment, de
zekerheid en de kwaliteit, en beslist of het koopt/houdt/verkoopt.

## Bestanden

```
impact_agent/
├── README.md                    ← dit bestand
├── skill/
│   └── impact-agent-skill.md    ← de skill (prompt) voor je eigen LLM
├── config.example.yaml          ← voorbeeld-config
├── data/
│   └── events.jsonl             ← voorbeeld-trainingsdata
└── scripts/
    └── finetune_impact.py       ← startpunt voor fine-tuning
```
