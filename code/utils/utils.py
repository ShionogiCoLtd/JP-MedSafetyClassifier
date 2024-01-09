from typing import Dict, Union
import math
import re
from pathlib import Path
import datetime as dt

import pandas as pd
import numpy as np

from .external_args import DataArguments
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

def validate_csv_after_load(
        df: pd.DataFrame,
        data_args: DataArguments,
    ) -> pd.DataFrame:
    """
    config.ymlで設定した列名が存在するかどうかを確認する
    存在しない場合は、入力値がblankの列を強制的に作成して表示エラーを回避する
    """
    re_strings = [
        'sentence[12]_key',
        'col_title',
        'col_drugname',
    ]
    for attr_name in dir(data_args):
        attr_value = getattr(data_args, attr_name)
        if re.match('^{}$'.format('|'.join(re_strings)), attr_name):
            if attr_value not in df.columns:
                df[attr_value] = ''
        if attr_name == 'col_id':
            if attr_value not in df.columns:
                df[attr_value] = np.arange(len(df))
        if attr_name == 'col_date':
            if attr_value not in df.columns:
                df[attr_value] = dt.datetime.now()
        if re.match('^col_output_[A-Za-z]+$', attr_name):
            if attr_value not in df.columns:
                df[attr_value] = None
    df['datetime'] = df[data_args.col_date].apply(pd.to_datetime).dt.strftime('%Y%m%d')
    df['guid'] =  df['datetime'].astype(str) + \
                        '_' + df[data_args.col_id].astype(str) 
    return df

# ゲージの予測値を調整
def get_modified_value(value, threshold):
    """
    ゲージの予測値を視覚的に見やすい値に変換する：
    元々の値は0～1、分類閾値も0.5ではない値を設定している
    それらを均して-1.0～0.0と0.0～1.0の範囲で表示させるように処理を行う
    """
    if value >= threshold:
        diff = value - threshold
        mod_v = diff/(1-threshold)
    else:
        diff = threshold - value
        mod_v = diff/threshold * -1
    return mod_v

def create_dataset(
        df: pd.DataFrame,
        data_args: DataArguments,
    ) -> Dataset:
    def create_dataset_label(df: pd.DataFrame):
        # 予測時はダミーの値を入れてエラーを回避させておく
        df[data_args.features_label] = np.zeros((len(df), 4)).tolist()
        df[data_args.col_title].replace(np.nan, '', inplace=True)
        df[data_args.sentence1_key].replace(np.nan, '', inplace=True)
        df[data_args.sentence2_key].replace(np.nan, '', inplace=True)
        return df
    
    def remove_cols(df: pd.DataFrame):
        rm_columns = [col for col in df.columns if col not in ['guid', text_a, text_b, label]]
        df.drop(rm_columns, axis=1, inplace=True)
        return df
        
    text_a = data_args.sentence1_key
    text_b = data_args.sentence2_key
    label = data_args.features_label
    
    df_test = remove_cols(create_dataset_label(df))
    
    # Create dataset
    datasets = Dataset.from_pandas(df_test)
    
    return datasets

def preprocess_function(
        example: Dict,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        padding: Union[str, bool],
        data_args: DataArguments,
    ):
    # Tokenize the texts
    def truncation_method(tokens, truncation_target, target_length):
        """
        # 入力テキスト長の調整
        [CLS][SEP][SEP]以外のtoken長が510以上のときに509に収まるように切り捨てを行う
        切り捨てる場所はiniファイルで指定する
        head、body（残るのはhead 1/3, tail 2/3）。tail
        text_a, text_bが存在するときに合計長が510以上の場合、
        1. 両方とも255以上: text_aを254、text_bを255に収まるように処理
        2. 一方が254以下でもう一方が256以上: 長い方を255に収まるように処理
        """
        if truncation_target == 'head': # 文頭切り捨て
            return tokens[-target_length:]
        elif truncation_target == 'tail': # 文末切り捨て
            return tokens[:target_length]
        elif truncation_target == 'body': # 文中央切り捨て
            return tokens[:math.floor(target_length * 1/3)] + \
                    tokens[-math.ceil(target_length * 2/3):]
        else:
            raise ValueError(truncation_target)
    sentence1_key = data_args.sentence1_key
    sentence2_key = data_args.sentence2_key
    truncation_target = data_args.truncation_target

    text_a = tokenizer.tokenize(example[sentence1_key])
    len_text_a = len(text_a)
    if example[sentence2_key]:
        text_b = tokenizer.tokenize(example[sentence2_key])                            

        # max_seq_lengthで2つの文章のtruncation量を調整
        len_text_b = len(text_b)
        if len_text_a + len_text_b > max_seq_length - 3:
            if (len_text_a >= int(max_seq_length/2) - 1
                and len_text_b >= int(max_seq_length/2) - 1
                ):
                text_a = truncation_method(text_a, 
                                            truncation_target,
                                            int(max_seq_length/2) - 2)
                text_b = truncation_method(text_b, 
                                            truncation_target,
                                            int(max_seq_length/2) - 2)
            elif len_text_a > len_text_b:
                text_a = truncation_method(text_a, 
                                            truncation_target,
                                            int(max_seq_length/2) - 1)
            else:
                text_b = truncation_method(text_b, 
                                            truncation_target,
                                            int(max_seq_length/2) - 1)
    else:
        text_b = example[sentence2_key]
        if len_text_a > max_seq_length - 3:
            text_a = truncation_method(text_a, 
                                        truncation_target,
                                        max_seq_length - 3)
            
    if text_a == []:
        text_a = ''
    if text_b == []:
        text_b = ''
    try:       
        result = tokenizer.encode_plus(text=text_a,
                                        text_pair=text_b, 
                                        max_length=max_seq_length,
                                        truncation=True,
                                        padding=padding)
    except:
        raise ValueError((example[sentence1_key],
                            example[sentence2_key],
                            text_a,
                            text_b))

    # Map labels to IDs (not necessary for GLUE tasks)
    if "label" in example:
        feature_label = "label"
    elif data_args.features_label:
        feature_label = data_args.features_label
    else:
        raise ValueError("check your feature_label name")
    result["label"] = example[feature_label]
    return result
