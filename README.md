# Tranformer-Based Classification of Astronomical Light Curves

> Bachelor's thesis of Alexander Mateides (2026)

### Abstract

This bachelor’s thesis investigates the use of transformer-based neural networks for the supervised classification of variable stars from high-precision
light curves obtained by the TESS satellite. Variable-star classification is an
important task in modern astronomy, as the growing volume of photometric
data from large-scale sky surveys makes manual analysis impractical.

While conventional machine-learning methods and convolutional neural
networks have already been applied successfully to this problem, transformer
architectures offer a promising alternative due to their ability to capture both
local and long-range dependencies in time series.

The thesis focuses on the design, implementation, and evaluation of a
transformer-based framework for distinguishing between selected classes of
eclipsing, pulsating, and rotating variable stars. The work includes an overview
of existing pretrained transformer architectures suitable for time-series analysis, selection of an appropriate model and dataset, identification of a manually
classified sample of TESS light curves, and development of a preprocessing
pipeline covering time-series standardization, detrending, normalization, and
input representation. 

Multiple experiments are carried out using different
model configurations and hyperparameter settings, and the resulting performance is assessed with standard multi-class classification metrics. The proposed approach is further compared with a simple baseline model and with
classifications available in published astronomical catalogues.

In addition to predictive accuracy, the thesis also evaluates the compu-
tational demands and practical suitability of transformer-based methods for
large-scale variable-star classification, and discusses their limitations and possible directions for future improvement.

### Acknowledgment

<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).

### License

This project is published under MIT license, however it uses several open-source libraries and you must their copyright and/or license when redistributing our software.

More information [here](THIRD_PARTY_LICENSES.md).
