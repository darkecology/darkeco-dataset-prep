# darkeco-dataset-prep

Data preparation for the [Dark Ecology Dataset](https://github.com/darkecology/darkeco-dataset):

* [`scripts/`](scripts/): process vertical profiles of biological activity (output from the [`cajun`](https://github.com/darkecology/wsrlib/blob/master/src/radar/cajun.m) function in [wsrlib](https://github.com/darkecology/wsrlib)) into summary datasets at scan, 5-minute, and daily temporal resolutions across ~160 NEXRAD stations since 1995.
* [`validation/`](validation/): perform validation and reproduce plots and validation outputs in our forthcoming paper about the dataset (see [suggested citation](https://github.com/darkecology/darkeco-dataset#citation)).
* [`schemas/`](schemas/): produce [Frictionless Table Schemas](https://specs.frictionlessdata.io//table-schema/) for our data.

Find more information about the Dark Ecology Project at [this link](https://darkecology.github.io/).
