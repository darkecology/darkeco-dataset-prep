# darkeco-dataset-prep

Data preparation pipeline for the [Dark Ecology](https://github.com/darkecology) project. Processes raw NEXRAD radar profiles (output from the [`cajun`](https://github.com/darkecology/wsrlib/blob/master/src/radar/cajun.m) function in [wsrlib](https://github.com/darkecology/wsrlib)) into analysis-ready summary datasets at scan, 5-minute, and daily temporal resolutions across ~160 NEXRAD stations since 1995.

The resulting dataset is published at [darkeco-dataset](https://github.com/darkecology/darkeco-dataset).

See `scripts/README.md` for pipeline details.
