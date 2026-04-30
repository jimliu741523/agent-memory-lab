# agent-memory-lab recall bench

- Turns: 97
- Target fact injected at turn 3 (user message)
- Deterministic, stdlib-only, no network

| pattern | recall (turn 3 fact) | final-context chars | extra callback calls | notes |
|---|---|---|---|---|
| sliding_window | no | 303 | 0 | window=10 |
| summary_compression | no | 1451 | 7 | trigger=20 keep=10 (mock summarizer) |
| hierarchical_summary | no | 225 | 13 | leaf=10 fanout=3 keep_recent=5 (mock) |
| vector_retrieval | yes | 303 | 87 | keep_recent=10 hash-BOW embed, recall via query() |
| structured_episodic | yes | 303 | 0 | keep_recent=10 recall via recall_episodes() |
