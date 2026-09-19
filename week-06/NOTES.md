# Notes

**Behavioral difference (ADK vs. LangGraph), same prompt:**
*"How many total hours should I plan to review everything required before week 9?"*

- **ADK:** recursively called `get_week_prerequisites` on each week it got back, walking the
  full chain (9 → 8 → 6, 7 → 1) → total **14 hours**.
- **LangGraph:** stopped after one hop, only expanding week 9's direct prerequisite (week 8) →
  total **11 hours**.
