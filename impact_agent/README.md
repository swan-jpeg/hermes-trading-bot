# Impact Agent — Skill + LLM-tutorial

De Impact Agent is een **tussenlaag** die gebeurtenissen interpreteert en
bepaalt welke instrumenten (stocks, obligaties, ETF's) geraakt worden — en met
welk sentiment. Dit is de schakel tussen een gebeurtenis en het RL-model.

> **"Trump zegt dat Jensen Huang een goede gozer is en je hem kan vertrouwen"**
> → de agent begrijpt: dit is **positief voor NVIDIA (NVDA)**, sentiment +0.8.

## Wat je hier vindt

```
impact_agent/
├── skill/
│   └── impact-agent-skill.md   ← de SKILL: zet dit op je eigen LLM
└── README.md                   ← deze tutorial: hoe je het implementeert
```

## De skill

`skill/impact-agent-skill.md` is een **system-prompt** die je op je eigen LLM
zet (Qwen3-27B, Llama-3.1, of elk groot model). Het geeft de LLM de rol van
impact-analist met:

- **Output-formaat**: JSON-vector per instrument
  `{entity, asset_class, sentiment, confidence, reason}`
- **Regels**: context begrijpen (niet alleen keywords), sentiment -1..+1,
  alleen beïnvloede instrumenten, `[]` als er geen impact is.
- **Voorbeelden**: Trump→NVDA, overheidsuitgaven→AGG/XLI, geen impact→`[]`.

## Hoe je het implementeert

### Stap 1 — Zet de skill op je LLM
Kopieer de inhoud van `skill/impact-agent-skill.md` als system-prompt in je
LLM (of als skill in je agent-framework). Je LLM functioneert nu als impact
agent.

### Stap 2 — Sluit je LLM aan op de bot
De bot stuurt elke gebeurtenis naar je LLM en krijgt de beïnvloede
instrumenten terug. Configureer in `.env` (secrets, nooit in GitHub):

```env
# Je eigen LLM als impact agent
IMPACT_LLM_API_KEY=<jouw key>
IMPACT_LLM_MODEL=qwen/qwen3-27b
IMPACT_LLM_BASE_URL=<jouw endpoint, bv. http://localhost:11434/v1>
```

En in `config/config.yaml`:
```yaml
impact_use_llm: true
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

### Fallback (geen LLM nodig)
Zonder API-key valt de impact agent terug op **keyword-matching** (in
`hermes_bot/impact.py`) — snel, gratis, offline. Werkt altijd.

## De output-vector

Elke beïnvloede entiteit krijgt een vector die het RL-model gebruikt:
```json
{
  "entity": "NVDA",
  "asset_class": "stock",
  "sentiment": 0.8,
  "confidence": 0.7,
  "reason": "Trump prijst Jensen Huang — positief voor AI-chips"
}
```

Het RL-model ziet per stock het sentiment, de zekerheid en de kwaliteit, en
beslist of het koopt/houdt/verkoopt.
