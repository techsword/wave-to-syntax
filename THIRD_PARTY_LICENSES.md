# Third-party licenses

This repository is distributed under the Apache License, Version 2.0 (see
`LICENSE`). It also contains or depends on the following third-party work.

## john-hewitt/structural-probes

- Source: https://github.com/john-hewitt/structural-probes
- License: Apache License, Version 2.0
- Copyright: Copyright 2019 John Hewitt
- Where:
  `src/spoken_syntax_probe/structural/model.py` and
  `src/spoken_syntax_probe/structural/regimen.py` are derived from this
  project, with modifications (adapted imports and flat-schema probe wiring
  for the spoken-language structural-probe experiments in this repository).
  Both file headers carry the derivation notice. The upstream note is
  preserved in `src/spoken_syntax_probe/structural/README-upstream.md`.

## ursa

- Source: https://github.com/gchrupala/ursa
- License: Apache License, Version 2.0
- Copyright: Copyright Grzegorz Chrupała
- Where: a pinned Git dependency,
  `ursa @ git+https://github.com/gchrupala/ursa@3dca2f1e68f312623d9e57e41aa9f8810fb0910f`,
  used for the tree-kernel and RSA analysis. It is not vendored in this
  repository.

## Universal Dependencies English-EWT data

`ewt.json` is adapted from the Universal Dependencies English-EWT treebank and
is distributed under CC BY-SA 4.0. See [`DATA_LICENSE.md`](DATA_LICENSE.md).
That data file is **not** covered by the repository's Apache-2.0 license.
