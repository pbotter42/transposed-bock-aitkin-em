# Release Checks

For `1.4.0rc1`, checked on macOS arm64 with Python 3.11:

- 33 tests pass, including all three fitted model specifications, frozen
  source regression comparisons, analytic derivatives, stopping behavior,
  numerical failures, invalid inputs, and command-line output.
- Ruff lint and formatting checks pass.
- Source distribution and wheel build successfully.
- All 32 saved reporting files match their manuscript-snapshot hashes.
- Public source and the archived reference package were checked for local
  machine paths and credential patterns. WILD responses are not included.

GitHub Actions is configured for Python 3.10, 3.11, and 3.12 on Ubuntu 24.04.
The release-time jobs could not start because GitHub reported an account
billing lock. There are therefore no passing hosted Linux results to claim
for this release. The workflow can be rerun after the account issue is
resolved. Local checks are not a substitute for that platform matrix.

These are software checks, not new production simulation replications,
new WILD estimates, or certification of the method's statistical validity.
