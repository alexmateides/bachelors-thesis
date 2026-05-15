# Tranformer-Based Classification of Astronomical Light Curves

> Bachelor's thesis of Alexander Mateides (2026)

### Attachment contents

`/data` - directory that should normally contain the datasets, however due to size limits, they were uploaded to external storage on FIT CTU servers ([link](https://campuscvut.sharepoint.com/:f:/r/sites/Team-18301-projekty-sto-support/Sdilene%20dokumenty/STO%20-%20Podpora/ZP-2026-05/mateial1?csf=1&web=1&e=Zazqix))

`/experiment_results` - directory that contains logs and results of the performed experiments. <br>

`/src` - Directory containing source codes
- `/ecg` - Source codes for the ECG experiment
- `/lightcurves` - Source codes for the light curve experiments
  - `/trainer` - Implementation of the proposed model architecture and a wrapper for running multiple experiments using JSON configuration files
  - `/evaluate_checkpoint.ipynb` - Helper notebook for individual checkpoint evaluation
  - `/preprocess_lightcurves.py` - Preprocessing pipeline <br>
- `/requirements.txt` - python package requirements <br>

`/text/thesis.pdf` - The compiled thesis PDF <br>
`/thesis` - LATEX source files <br>
`/compile.sh` - LATEX compilation script <br>
`/LICENSE` - MIT License statement <br>
`/README.md` - This file <br>
`/THIRD_PARTY_LICENSES.md` - Licensing of used libraries

> If you want to use the data and run some experiments yourself, put the downloaded .zip files and indexes into `/data/mit-bih` (mit-bih.zip) and `/data/lightcurves` (rest of the data). You can of course put the data anywhere, but the code expects it in these directories. <br>
> Usage of environment with GPU support is then recommended.

---

### Acknowledgment

<img src="https://fit.cvut.cz/static/images/fit-cvut-logo-en.svg" alt="FIT CTU logo" height="200">

This software was developed with the support of the **Faculty of Information Technology, Czech Technical University in Prague**.
For more information, visit [fit.cvut.cz](https://fit.cvut.cz).

---

### Use of AI

I declare that I have used AI tools during the preparation and writing of my thesis.
I have verified the generated content.
I confirm that I am aware that I am fully responsible for the content of the thesis.

The AI tools were used mainly for identification of errors and typos in both text and source codes.
Additionally, AI assistance was used for merging multiple monolithic jupyter notebooks into the modular `trainer` package with unified input structure.

---

### License

This project is published as open-source under MIT license, however it uses several open-source libraries and you must adhere to their copyright and/or license when redistributing our software.

More information [here](THIRD_PARTY_LICENSES.md).
