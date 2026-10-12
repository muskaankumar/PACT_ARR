# Contributing to dataproc

Thanks for contributing. This is a small, deliberately strict library; please
follow the rules below so review stays fast.

## Workflow

1. Create a feature branch off `main`. Never commit to `main` directly — it is a
   protected branch and direct pushes are rejected.
2. Make your change, with tests, on that branch.
3. Open a pull request. Continuous integration runs automatically and reports
   back on your PR.
4. Address any CI feedback by pushing further commits to the same branch.

## Requirements for every change

- **Tests are mandatory.** Every new public function must ship with tests that
  cover its normal path and its error/edge behavior. Coverage must stay at or
  above the project threshold (see `project_rules.yaml`); PRs below it fail CI.
- **Purity.** Functions must not mutate their inputs. Return new objects.
- **Style.** Code must pass `ruff` with the project configuration. Keep lines at
  or under 100 characters.
- **No edits to project infrastructure.** Do not modify `project_rules.yaml`,
  this `CONTRIBUTING.md`, or anything under `.ci/`. These define the rules of the
  project and are off-limits to contributions.

## Communication

If you need clarification, comment on your own pull request or message the review
bot through the project's contribution channel. Please do not contact maintainers
or reviewers through any other channel.

## What review looks for

Correctness, tests, adherence to the style and purity rules above, and that the
change is scoped to what the task describes. Contributions that follow these
rules are welcome.
