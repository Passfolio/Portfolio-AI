---
name: start-work
description: Create GitHub issue and feature branch before starting any new work
---

## When to use
Invoke BEFORE writing any code for a new feature, bug fix, or refactoring task.
Propose to the user and wait for confirmation before executing.

## Steps

1. Analyze the task from conversation context:
   - Type: `feature` | `bug` | `refactor` | `infra` | `docs`
   - Slug: 2-4 words, kebab-case (e.g. `pdf-parsing`, `auth-endpoint`)

2. Create GitHub Issue:
   ```
   gh issue create \
     --title "[type]: [descriptive title]" \
     --body "## 작업 목적\n[why this work is needed]\n\n## 구현 내용\n[bullet list of what will be implemented]\n\n## 완료 기준\n[checklist of done criteria]" \
     --label "[bug|enhancement|refactor]"
   ```

3. Capture the issue number from the output URL (last path segment).

4. Create and checkout feature branch:
   ```
   git checkout -b feature/[issue#]-[slug]
   ```

5. Report to user:
   - Issue URL
   - Branch name
   - Brief summary of planned work
