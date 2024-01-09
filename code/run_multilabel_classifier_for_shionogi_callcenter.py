# 20230818
# - 公開のために各種設定を外出し
import logging
import sys
from typing import Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils.logging import (
    set_verbosity_info,
    enable_default_handler,
    enable_explicit_format,
)
from optimum.bettertransformer import BetterTransformer

from utils import (
    DataArguments,
    ModelArguments,
    create_dataset,
    preprocess_function,
)

def ml_text_classifier(
        df: pd.DataFrame,
        args: Tuple[ModelArguments, DataArguments, TrainingArguments],
        progress_function=None,
        logger=None,
    ):
    model_args, data_args, training_args = args
    if not logger:
        # Setup logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,   # DEBUG, INFO, WARNING, ERROR, CRITICAL
        )
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    set_verbosity_info()
    enable_default_handler()
    enable_explicit_format()
    logger.debug(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Create evaluation data.
    datasets = create_dataset(df, data_args)
    logger.info(datasets)
    
    # Labels
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = ['問合せ：副作用', 
                  '問合せ：妊婦・授乳婦服薬',
                  '問合せ：SS',
                  '問合せ：AllNegative',
                 ]
    num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
  
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(training_args.device)
    model = BetterTransformer.transform(model)

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    datasets = datasets.map(lambda example: preprocess_function(
                                        example=example,
                                        tokenizer=tokenizer,
                                        max_seq_length=max_seq_length,
                                        padding=padding,
                                        data_args=data_args,
                                        ), batched=False,
                                        desc='Preprocess the data:',
                                        num_proc=data_args.num_workers_of_preprocess,
    )
    # Filter columns
    rm_columns = [col for col in datasets.column_names if col not in ['guid', 'input_ids',
                                                            'token_type_ids', 'attention_mask', 'label']]
    datasets = datasets.remove_columns(rm_columns)

    predict_dataset = datasets

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    # 予測結果の取得
    data_loader = DataLoader(
                        dataset=predict_dataset.remove_columns('guid'),
                        batch_size=training_args.per_device_eval_batch_size,
                        shuffle=False,
                        collate_fn=data_collator,
                        num_workers=data_args.num_workers_of_data_loader,
    )
    total_iterations = len(data_loader)
    all_results = []
    logger.info(f'Inference Start: {len(predict_dataset)} records in {total_iterations} iterations.')
    for i, batch in enumerate(tqdm(data_loader, desc='Predict Examples')):
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            all_results.append(logits.cpu().numpy())
        if progress_function:
            progress_function((i + 1, total_iterations))
    result = np.concatenate(all_results, axis=0)
    logger.info(f'Inference Finished.')


    def sigmoid(a):
        return 1 / (1 + np.exp(-a))
    guids = predict_dataset['guid']
    predictions = sigmoid(result)
    df_preds = pd.DataFrame(predictions,
                            columns=[f'pred_{i}' for i in ['ADR', 'pregnancy', 'SS', 'allnegative']])
    df_preds['guid'] = guids
    return df_preds

def main():
    import yaml
    from utils import (
        get_ext_args,
        validate_csv_after_load,
    )

    # data_path = '/PATH/TO/YOUR/CSV'
    data_path = '/home/wada/workspace/shionogi_dashboard/release/20230817/data/20210201_問合せ_ダミー_no_inputs.csv'
    baseDir = Path(__file__).parent.parent    
    with open(baseDir / 'config.yml', 'r', encoding='utf-8') as f:
        yml = yaml.load(f, yaml.FullLoader)
    args = get_ext_args(cfg=yml)
    model_args, data_args, _ = args
    model_args.model_name_or_path = str(baseDir.joinpath('model', model_args.model_name_or_path))
    df = pd.read_csv(data_path, encoding='cp932')
    df = validate_csv_after_load(df, data_args)
    results = ml_text_classifier(df, args)
    print(results)
    return
    
if __name__ == '__main__':
    main()