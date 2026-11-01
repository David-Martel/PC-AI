# Coverage upload authentication

At PC_AI head `62a1d7d3f0b1d89e70b86dad895ceb3191cca113`, CI run 38042534328 attempt 2 completed 2,602 PowerShell tests with zero failures, 39 skips and zero NotRun. The unchanged 85% coverage gate failed at 13,561/21,145 distinct commands (64.1334%). This coverage debt remains separate from provider delivery.

The same job found the coverage report but Codecov rejected its protected-branch upload because authentication was missing. The action step still reported success because `fail_ci_if_error` was false. Passing that step did not establish receipt of the report.

The maintained workflow now requests GitHub OIDC authentication through the existing pinned Codecov action. Only the two coverage-producing jobs receive `id-token: write` and `contents: read`; other jobs receive no new permissions. Upload errors now fail explicitly. The official hash-pinned CLI, action revision, report paths, coverage denominator and 85% requirement are unchanged. No upload token is stored or added to GitHub or Bitwarden.

Sources reviewed: the exact action revision [303a32d7](https://github.com/codecov/codecov-action/tree/303a32d7a59b442fa8d48b6a1cc6825c09c847a5), its `use_oidc` input and audience selection, and [GitHub's OIDC reference](https://docs.github.com/en/actions/reference/security/oidc). The pinned action skips OIDC for fork pull requests and retains its public fork behavior. A successful real upload is required before provider delivery is accepted; syntax or action-source review alone is insufficient.

Source review and local workflow syntax validation passed: ten YAML files and 82 embedded PowerShell steps parse, and the permission and input changes are limited to the two coverage jobs. Authentication delivery remains **NOT_TESTED** until CI at the published successor head completes a real upload. Existing test passes do not establish real Redis, native runtime, consumer deployment, performance or HIL acceptance.
