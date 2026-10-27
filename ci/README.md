# CI workflow

`build.yml` here is the real GitHub Actions workflow. It lives outside
`.github/workflows/` only because the token used to push this repo does not
carry the `workflow` scope, and GitHub rejects any push that touches a workflow
file without it.

To activate it:

```bash
gh auth refresh -s workflow          # one-time, grants the missing scope
mkdir -p .github/workflows
git mv ci/build.yml .github/workflows/build.yml
git commit -m "Enable the cross-platform CI workflow" && git push
```

Or paste the file straight into the GitHub web UI at
`.github/workflows/build.yml` — the browser session already has the scope.
