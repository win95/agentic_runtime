---
name: data-access
description: Access the three data stores for an asset — Postgres faults, maintenance and sensors, Neo4j asset relations, Qdrant technical documents — by following a map; gather evidence through tools, never invent data.
keywords: data access postgres qdrant neo4j retrieve documents relations graph store asset
map:
  start: identify
  nodes:
    - id: identify
      goal: Parse asset_id and date range from the input; validate the asset_id format.
      tools: []
      stop_when: asset_id missing or invalid — stop and report it.
    - id: postgres-faults
      goal: Search fault history for the asset in the date range.
      tools: [search_fault_history]
      branches:
        - when: no faults found — do not stop, continue to maintenance
          goto: postgres-maintenance
    - id: postgres-maintenance
      goal: Get maintenance records for the asset.
      tools: [get_maintenance_records]
    - id: postgres-sensors
      goal: Get sensor readings for the asset in the date range to correlate with faults.
      tools: [get_sensor_data]
      branches:
        - when: faults and maintenance are both empty (nothing to correlate)
          goto: graph-relations
    - id: graph-relations
      goal: Query Neo4j for the asset's relations (building, components); note any component that recurs in the fault evidence for synthesis.
      tools: [query_asset_relationships]
    - id: qdrant-documents
      goal: Search technical documents using the fault descriptions as the query; when there are no faults, query with asset + component keywords.
      tools: [search_technical_documents]
    - id: synthesize
      goal: Draft findings only from the collected evidence.
      tools: []
      stop_when: all three stores unreachable (error evidence only) — stop and report plainly; or enough evidence collected.
---

# Data Access

## When to Use
- The mission needs data about one asset from the platform's stores: Postgres (faults, maintenance, sensors), Neo4j (asset relations), Qdrant (technical documents). E.g. mission `fault_analysis` with `asset_id` + date range in the input.

## Map
Follow the `map:` in this file's frontmatter as a plan: start at `identify`, then Postgres (faults → maintenance → sensors), Neo4j relations, Qdrant documents, and `synthesize`. The default flow is the next node in order; take a branch when its condition holds. You may deviate when the evidence justifies it, but say why, and report the nodes you visited at the end.

## Rules
- Never invent data. If a store is unreachable, the tool returns error evidence — report the failure plainly and keep going; error evidence is still evidence.
- Only query the asset named in the input. `query_asset_relationships` reads Neo4j; `get_sensor_data` reads real rows from Postgres (`sensor_readings`) — an empty list means no readings in the range.
- Draft findings ONLY from evidence actually collected. If all three stores are unreachable, the honest answer is to say so, not a guessed finding.

## What NOT to Do
- Do not guess a finding when all evidence is errors.
- Do not present an error message as if it were a data finding.
- Do not call tools for assets other than the one in the input.
