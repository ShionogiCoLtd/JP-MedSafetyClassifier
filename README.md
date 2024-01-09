# JP-MedSafetyClassifier

## Overview
JP-MedSafetyClassifier is a BERT-based deep learning model developed for the classification of medical safety events in Japanese. It wraps the model in a Dash web application for visualization of output results and assists in file output through user input.

## Main Features
The JP-MedSafetyClassifier can classify into three main categories:
1. **Adverse Events**: Positive classification for adverse events caused by the use of a company's pharmaceutical products.
2. **Pregnant/Nursing Medication**: Cases involving medication use during pregnancy or nursing. Inquiries about prescriptions without confirmation of administration are classified as negative.
3. **Special Situation (SS)**: Refers to non-standard medical situations, important for clinical aspects or drug safety, recommended for reporting in regulatory guidelines like ICH, FDA, EMA. Includes overdose, drug abuse/misuse, off-label use, occupational exposure, insufficient effect, and administration errors.

## Usage
1. Modify `config.yml` according to your data column names and the location of the pre-trained model.
2. Execute `python ./code/app.py` to launch the Dash web application.
3. Access the application through a browser at `http://127.0.0.1:1234/`.

Example `config.yml`:

```yaml
# Set encoding to "UTF-8" when editing and saving with a text editor.

# Dash Settings
app_setting:
  debug_mode: True
  host: 127.0.0.1
  port: 1234
  tab_disabled:
    filter: False
    threshold: True
  download_default_type: ".xlsx"

# Data handling, preprocessing, and output column definitions
data:
  sentence1_key: ...
  sentence2_key: ...
  col_title: ...
  col_id: ...
  col_date: ...
  col_drugname: ...
  col_output_adr: ...
  col_output_pregnancy: ...
  col_output_ss: ...
  truncation_target: body

# Prediction Model Settings
transformers:
  model_name_or_path: model_release.20230817
  output_dir: ./tmp
  overwrite_cache: False
  use_fast_tokenizer: False
  max_seq_length: 512
  per_device_eval_batch_size: 8
  num_workers_of_preprocess: 1
  num_workers_of_data_loader: 2
