---
name: smart-commit
description: Group related file changes and commit with conventional commit messages. Runs automatically after each logical unit of work — no user confirmation needed.
---

## When to use
Run automatically (no confirmation needed) after completing a logical unit:
- Data model / schema implementation
- Service or business logic
- API route / endpoint
- Test additions
- Config or dependency changes

## Steps

1. `git status` — see all modified/new/deleted files.

2. Group files by logical purpose:

   | Group   | Files                                          |
   |---------|------------------------------------------------|
   | models  | models/, schemas.py, db.py                     |
   | services| services/*.py                                  |
   | api     | api/, routes/, endpoints/                      |
   | tests   | tests/, *_test.py, test_*.py                   |
   | config  | config.py, .env*, requirements*, alembic/      |
   | docs    | *.md, docstrings                               |

3. Conventional commit types:
   - `feat`: new functionality
   - `fix`: bug fix
   - `refactor`: restructure without behavior change
   - `test`: test additions/changes
   - `docs`: documentation
   - `chore`: config, deps, build

4. For each non-empty group:
   ```
   git add [specific files in group]
   git commit -m "[type]([scope]): [short imperative description in English]"
   ```

5. Do NOT push. Do NOT ask for confirmation.

6. After all commits: print a one-line summary per commit made.
