---
name: fault-investigation
description: Investigate equipment faults step by step — gather fault history, maintenance records, technical documents and asset relations in order, never invent data, and draft root causes only from fault and maintenance evidence.
keywords: fault equipment elevator failure breakdown asset maintenance sensor motor root
---

# Fault Investigation

## When to Use
- The mission is to analyze equipment faults and identify root causes (e.g. mission `fault_analysis`, with `asset_id` + date range in the input).

## Steps
1. Gather evidence in this order: fault history (`search_fault_history`), maintenance records (`get_maintenance_records`), technical documents (`search_technical_documents`), asset relations (`query_asset_relationships`).
2. Never invent data. If a store is unreachable, the tool returns error evidence (`kind: "error"`) — report the failure plainly and keep going; error evidence is still evidence.
3. `get_sensor_data` is a stub (it returns `"stub": true`) — never present stub readings as real measurements.
4. Draft `root_causes` ONLY from evidence with `kind` "fault" or "maintenance". If there is none, the honest answer is "unknown — insufficient evidence", not a guessed cause.
5. Recommendations must follow from the evidence actually collected.

## What NOT to Do
- Do not guess a root cause when all evidence is errors.
- Do not present an error message as if it were a fault finding.
- Do not call tools for assets other than the one in the input.
