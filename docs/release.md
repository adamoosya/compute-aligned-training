# Release checklist

## Local final check

```bash
python -m pytest -q
python scripts/check_release.py
python scripts/build_paper_assets.py --output outputs/paper-assets-final
```

The preflight checks tracked and untracked nonignored candidates, syntax,
configuration parsing, reference hashes, private-path/credential patterns and
large files. It is a limited pattern scan, not an exhaustive secret audit or
Git-history audit. It never displays matched credential values.

The maintainer selected MIT licensing for the code and documentation and
`adamoosya/compute-aligned-training` as the repository destination. Initial upload
is private. The LICENSE and CITATION.cff record those choices; adding this
metadata does not create the remote repository or change its visibility.
Confirm redistribution terms for the included numerical reference files before
public release. Third-party models and datasets retain their own terms.

An access token was found in a private historical source file. That file is not
included in this refactor. Rotate/revoke that old credential; simply omitting
it from this repository does not invalidate an exposed token.

## Commit and publish

Review files with `git status` and `git diff --cached --stat` after deliberately
staging the release set. Do not add raw archives, old scripts, credentials,
checkpoints, or outputs. GitHub Desktop may review the same repository, but
publishing can be done entirely from the terminal. Choose the owner/repository
and visibility before creating a remote. No provided installer publishes or
creates a remote automatically.

Run the generated GitHub Actions checks on the first push before making the
paper release. Its job runs CPU tests, not expensive model jobs or automatic
PyPI publication. GPU execution must remain labelled unvalidated until checked
on the target hardware. Full retraining is a separate reproducibility claim.

The final publication check is `python scripts/check_release.py --strict`;
this also requires a license and repository URL. It does not certify that all
scientific results or legal permissions have been verified.

## Reference documentation

- [GitHub: licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [GitHub: citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [GitHub: Python tests in Actions](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
