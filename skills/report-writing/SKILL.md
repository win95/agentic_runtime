---
name: report-writing
description: Write the mission result — answer, actions taken and evidence — citing the evidence relied on and marking unknown results honestly when the evidence is all errors.
keywords: report summary recommendation conclusion evidence write
---

# Report Writing

## When to Use
- Every mission run that must produce a result: fill `answer`, `actions_taken` and `evidence` from what actually happened in the run.

## Steps
1. `answer`: one concise summary of the findings. Admin missions answer in Vietnamese.
2. `actions_taken`: the tool names called during the run, in order.
3. `evidence`: the evidence items the answer relies on. Cite them — do not cite evidence you did not use.
4. If all evidence is errors (`kind: "error"`), say plainly that the run could not establish the finding and list what failed. Never present "unknown — insufficient evidence" as a finding.

## What NOT to Do
- Do not claim a root cause, a status, or a number that no evidence supports.
- Do not hide tool errors from the answer.
- Do not add recommendations that do not follow from the evidence.
