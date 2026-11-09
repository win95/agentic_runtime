# Skills — task playbooks for the Deep Agent

Skills are markdown playbooks the Deep Agent loads when the current task
matches, before calling tools. This is our self-hosted equivalent of the
"skills" piece in LangSmith Managed Deep Agents — no LangSmith involved.

## Format

One directory per skill under `skills/`, with a `SKILL.md` file:

```markdown
---
name: fault-investigation
description: What the skill does (shown to the agent when it decides
  whether the skill applies; keep it under 1024 characters).
keywords: fault equipment elevator failure
---

# Fault Investigation

## When to Use
...

## Steps
...

## What NOT to Do
...
```

Frontmatter fields: `name` (lowercase letters, digits, hyphens, max 64
chars), `description` (required — the trigger text the agent sees),
`keywords` (a space/comma-separated string or a YAML list — the trigger
terms our deterministic matcher uses). Unknown frontmatter keys are
ignored by the loader. The body after the frontmatter is the playbook.

## How selection works

`runtime/skills/loader.py` — deterministic, no LLM call just to pick
skills, logged at debug level with per-skill scores:

1. **Explicit declaration wins.** A mission listing skills in
   `execution.skills` (seed YAML) or `mission["skills"]` gets exactly
   those skills; unknown names are skipped with a warning.
2. **Otherwise keyword matching.** A skill is selected when at least 1
   of its distinct keywords appears (singularized, whole-word) in the
   mission id + objective + capabilities + request text. Ties break by
   score, then name.

## Delivery (native deepagents mechanism)

`deep_agent_node` passes the selected skills to
`create_deep_agent(skills=["/skills/"])`, which adds the native
`SkillsMiddleware` (deepagents 0.7.34) over the default in-memory
`StateBackend`. The selected `SKILL.md` contents are handed to the agent
via `invoke(files={"/skills/<name>/SKILL.md": {...}})`; the middleware
progressively discloses them (skill index in the system prompt, full text
on read). No real filesystem access is granted — the default backend
stays in-memory. A run with no matching skill behaves exactly as before
(no `skills=` argument, no `files` in the invoke input).

## Seeded skills

| Skill | When it applies |
| --- | --- |
| `fault-investigation` | Equipment fault analysis (`fault_analysis` mission): evidence order, never invent data, root causes only from fault/maintenance evidence. |
| `admin-operations` | Platform administration (`system_admin` mission): read before write, exact id for destructive actions, 409 delete rule. |
| `report-writing` | Every run that must produce a result: answer + actions_taken + evidence, cite what was used, honest "unknown" when evidence is all errors. |

To add a skill, create `skills/<name>/SKILL.md` in the format above —
no code change needed. To pin skills to a mission, add
`skills: [<name>, ...]` under `execution:` in the mission seed YAML.
