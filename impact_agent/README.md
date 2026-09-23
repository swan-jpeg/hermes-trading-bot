# Impact Agent — LLM-training & integratie

Deze map bevat alles om de **Impact Agent** te trainen en te integreren: een
gratis AI-model dat gebeurtenissen interpreteert en bepaalt welke instrumenten
(stocks, obligaties, ETF's) geraakt worden.

## Wat de Impact Agent doet

De Impact Agent is de schakel die een gebeurtenis koppelt aan beïnvloede
instrumenten. Voorbeeld:

> "Trump zegt dat Jensen Huang een goede gozer is en je hem kan vertrouwen"
>
> → De agent begrijpt: dit is **positief voor NVIDIA (NVDA)**.

Zonder deze agent weet het RL-model niet welke stocks geraakt worden door een
speech of rapport. De agent levert die koppeling.

## Hoe het werkt

```
Gebeurtenis (speech/rapport/overheidsuitgave/nieuws)
        ↓
IMPACT-AGENT  ← bepaalt: "dit raakt NVDA, TSLA, AGG"
        ↓
per beïnvloede entiteit: fusion-input (sentiment/zekerheid/kwaliteit)
        ↓
RL-model (per entiteit: welk instrument + gewicht + zekerheid)
        ↓
risk engine + Monte Carlo → portfolio → executie
```

## Twee modi

### 1. Keyword-matching (altijd, geen API-key nodig)
`ImpactAgent` in `hermes_bot/impact.py` matcht keywords/sectoren en koppelt die
aan beïnvloede tickers. Snel, gratis, offline. Werkt altijd.

### 2. LLM-interpretatie (optioneel, gratis model)
`LLMImpactAgent` gebruikt een gratis AI-model (via OpenRouter) om de **context**
van de gebeurtenis te begrijpen — vangt gevallen die keyword-matching mist
(bv. "Trump prijst Jensen Huang" → NVDA).

## Installatie & integratie

### Stap 1 — API-key (gratis)
1. Maak een gratis account op https://openrouter.ai
2. Genereer een API-key (Settings → Keys)
3. Zet de key in `.env` (NIET in GitHub):
   ```
   IMPACT_LLM_API_KEY=sk-or-...
   IMPACT_LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
   ```

### Stap 2 — Activeer de LLM in config
In `config/config.yaml`:
```yaml
impact_use_llm: true
impact_llm_model: meta-llama/llama-3.1-8b-instruct:free
```

### Stap 3 — Test
```bash
uv run python -c "
from hermes_bot.impact import LLMImpactAgent
agent = LLMImpactAgent()
r = agent.analyze('Trump says Jensen Huang is a good guy, you can trust him',
                  source='speech', entity_id='trump')
print([i['entity'] for i in r.impacted])
"
```

## Het model trainen/verfijnen (optioneel)

De gratis OpenRouter-modellen werken direct zonder training. Wil je een
**eigen** impact-model trainen (bv. op jouw historische gebeurtenissen), dan:

1. Verzamel data: `data/events.jsonl` met `{event, source, impacted_tickers, sentiment}`
2. Gebruik een fine-tuning-API (OpenRouter / Nous) op een open model
   (bv. Llama-3.1-8B).
3. Vervang `IMPACT_LLM_MODEL` door je getrainde model-id.

Zie `scripts/finetune_impact.py` voor een startpunt.

## Skills & MCP (voor de coder consultant)

De coder consultant kan deze skills/MCP gebruiken om de impact-agent te
verbeteren:

- **context7 MCP** — up-to-date docs voor LLM-API's (OpenRouter, Nous).
- **trading-bot-architecture skill** — de architectuurregels (impact-agent
  moet per-entiteit fusion-inputs leveren, RL stelt voor / Risk beslist).

## Bestanden

```
impact_agent/
├── README.md              ← dit bestand
├── data/                  ← trainingsdata (events.jsonl)
├── scripts/
│   └── finetune_impact.py ← startpunt voor fine-tuning
└── config.example.yaml    ← voorbeeld-config
```
