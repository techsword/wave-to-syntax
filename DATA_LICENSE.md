# Data license — `ewt.json`

`ewt.json` is **not** covered by this repository's Apache License, Version 2.0.
It is distributed under the Creative Commons Attribution-ShareAlike 4.0
International license (CC BY-SA 4.0).

- Adapted from: Universal Dependencies English-EWT
  (https://github.com/UniversalDependencies/UD_English-EWT)
- Annotations: © 2013–2021 The Board of Trustees of the Leland Stanford Junior
  University
- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
- Changes: converted the UD treebank into a JSON dataset of constituency
  parses and derived dependency-distance labels for the structural-probe
  evaluation (the `ref` and `test` splits).

## Underlying text (LDC caveat)

The trees in `ewt.json` include the original English Web Treebank token
strings. The annotations are derived from the English Web Treebank
(LDC2012T13), whose source text is distributed under Linguistic Data
Consortium terms. This file redistributes only the derived annotations and
their token strings for research reproducibility; the underlying corpus is not
included.

## Citation

If you use `ewt.json`, cite the English Web Treebank annotations:

> Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S. R., Connor, M.,
> Bauer, J., & Manning, C. D. (2014). A Gold Standard Dependency Corpus for
> English. In *Proceedings of the Ninth International Conference on Language
> Resources and Evaluation (LREC-2014)*, pages 2897–2904.

and the Universal Dependencies project:

> de Marneffe, M.-C., Manning, C. D., Nivre, J., & Zeman, D. (2021).
> Universal Dependencies. *Computational Linguistics*, 47(2), 255–308.
