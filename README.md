# JP-MedSafetyClassifier

## Overview
JP-MedSafetyClassifier is a BERT-based deep learning model developed for the classification of medical safety events in Japanese. It wraps the model in a Dash web application for visualization of output results and assists in file output through user input.

## Main Features
The JP-MedSafetyClassifier can classify into three main categories:
1. **Adverse Events**: Positive classification for adverse events caused by the use of a company's pharmaceutical products.
2. **Medications During Pregnancy and Breastfeeding**: Cases involving medication use during pregnancy or nursing. Inquiries about prescriptions without confirmation of administration are classified as negative.
3. **Special Situation (SS)**: Refers to non-standard medical situations, important for clinical aspects or drug safety, recommended for reporting in regulatory guidelines like ICH, FDA, EMA. Includes overdose, drug abuse/misuse, off-label use, occupational exposure, insufficient effect, and administration errors.

## Usage
1. Modify `config.yml` according to your data column names and the location of the pre-trained model.
2. Execute `python ./code/app.py` to launch the Dash web application.
3. Access the application through a browser at `http://127.0.0.1:1234/`.

![Screen Shot](./image/screenshot01.png)
