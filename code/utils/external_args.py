# 2023/04/10
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)


@dataclass
class DataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # ADD
    sentence1_key: str = field(
        default=None,
        metadata={
            "help": "Set the key name for sentence1."
        },
    )
    sentence2_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Set the key name for sentence2."
        },
    )
    features_label: str = field(
        default='問合せ：MultiLabels',
        metadata={
            "help": "Set column_name for label."
        },
    )
    col_title: str = field(
        default=None,
        metadata={
            "help": "Set column_name for InquiryTitle."
        },
    )
    col_id: str = field(
        default=None,
        metadata={
            "help": "Set column_name for InquiryID."
        },
    )
    col_date: str = field(
        default=None,
        metadata={
            "help": "Set column_name for InquiryDate."
        },
    )
    col_drugname: str = field(
        default=None,
        metadata={
            "help": "Set column_name for InquiryDrugName."
        },
    )
    col_output_adr: str = field(
        default=None,
        metadata={
            "help": "Set column_name for OutputADR."
        },
    )
    col_output_pregnancy: str = field(
        default=None,
        metadata={
            "help": "Set column_name for OutputPregnancy."
        },
    )
    col_output_ss: str = field(
        default=None,
        metadata={
            "help": "Set column_name for OutputSS."
        },
    )
    truncation_target: Optional[str] = field(
        default='body',
        metadata={
            "help": "strategy for long sentence."
        },
    )
    num_workers_of_data_loader: Optional[int] = field(
        default=1,
        metadata={
            "help": "a number of subprocesses for data_loader."
        },
    )
    num_workers_of_preprocess: Optional[int] = field(
        default=1,
        metadata={
            "help": "a number of subprocesses for preprocessing."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

if torch.cuda.is_available():
    no_cuda = False
    use_fp16 = True
else:
    no_cuda = True
    use_fp16 = False

def get_ext_args(cfg: Dict):
    args={
        'do_train': False,
        'do_eval': False,
        'do_predict': True,
        'no_cuda': no_cuda,
        'fp16': use_fp16,
    }
    args.update(cfg['transformers'])
    args.update(cfg['data'])
    model_args, data_args, training_args = parser.parse_dict(args)
    return model_args, data_args, training_args

disp_data_def = dict(
        index_list=[0],
        display_index=0,
        text_display_setting=None,
        inquiry_title_options={'0': '問合せ：タイトル'},
        inquiry_title_value='0',
        inquiry_title_disabled=True,
        inquiry_date_id_drug_children='問合せ日時; {}, 問合せID; {}, 製品詳細名; {}.'.format(' - ', ' - ', ' - '),
        inquiry_text_srcDoc=' - ',
        response_text_srcDoc=' - ',
        switch_ADR_on=False,
        switch_pregnancy_on=False,
        switch_SS_on=False,
        diff_inf_and_input_ADR='予測: {} / 前回入力: {}'.format('-', '-'),
        diff_inf_and_input_pregnancy='予測: {} / 前回入力: {}'.format('-', '-'),
        diff_inf_and_input_SS='予測: {} / 前回入力: {}'.format('-', '-'),
        gauge_ADR_value=0.0,
        gauge_pregnancy_value=0.0,
        gauge_SS_value=0.0,
        button_reflect_disabled=True,
        button_move_to_previous_disabled=True,
        button_move_to_next_disabled=True,
        button_move_to_previous_popover_children='Previous', 
        button_move_to_next_popover_children='Next',
        button_download_disabled=True,
)