# 20230410 CallBackをdict形式で整理
# 20230818 各種設定を外出しして整理
# 20230821 処理状況のリアルタイム取得
import io
import sys
import json
import base64
import re
import logging
from datetime import datetime
from chardet import detect
from pathlib import Path
import yaml

import pandas as pd
import numpy as np

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    ctx,
    DiskcacheManager,
)
import dash_bootstrap_components as dbc
import dash_daq as daq
import diskcache

from run_multilabel_classifier_for_shionogi_callcenter import ml_text_classifier
from utils import (
    get_ext_args,
    validate_csv_after_load,
    get_modified_value,
    disp_data_def,
)

# 実行のrootディレクトリ
baseDir = Path(__file__).parent.parent

# ログ保存
logDir = baseDir.joinpath('log')
if not logDir.exists():
    logDir.mkdir()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # ロガー自体のレベルをDEBUGに設定

# 標準出力（コンソール）用ハンドラの設定
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

# ファイル用ハンドラの設定
file_handler = logging.FileHandler(logDir.joinpath('{}.log'.format(datetime.now().strftime('%Y%m%d'))))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

# ロガーにハンドラを追加
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# config.ymlを読み込む
logger.info('Read config.yml')
with open(baseDir / 'config.yml', 'r', encoding='utf-8') as f:
    yml = yaml.load(f, yaml.FullLoader)
app_setting = yml['app_setting']
args = get_ext_args(cfg=yml)
model_args, data_args, training_args = args

model_args.model_name_or_path = str(baseDir.joinpath('model', model_args.model_name_or_path))

t_list = ['youdenj'] + [f'{rec:.2f}' for rec in np.arange(0.9, 1, 0.01)]
# 閾値の取得
th_dict = {}
with open(Path(model_args.model_name_or_path) / 'eval_results.json', 'r', encoding='utf-8') as f:
    js = json.load(f)
for i in range(3):
    th_dict.update({i: {rec: js[f'eval_{i}_{rec}_optimal_threshold'] for rec in t_list}})

# 各閾値に対応するPrecision, Recallの取得
th_dropdown_menu_dict = {}
with open(Path(model_args.model_name_or_path) / 'test_results.json', 'r', encoding='utf-8') as f:
    js = json.load(f)
for i in range(3):
    th_dropdown_menu_dict[i] = {t_list[0]: '【推】Rec. {:.1f}%/ Prec. {:.1f}%'.format(
        js[f'predict_{i}_{t_list[0]}_Recall'] * 100,
        js[f'predict_{i}_{t_list[0]}_Precision'] * 100,
    )}
    th_dropdown_menu_dict[i].update({item: 'Rec. {:.1f}%/ Prec. {:.1f}%'.format(
        js[f'predict_{i}_{item}_Recall'] * 100,
        js[f'predict_{i}_{item}_Precision'] * 100,
    ) for item in t_list[1:]})

# User Setを定義
th_usr_set = []
th_usr_set.append({
        0: '0.90',
        1: '0.90',
        2: '0.90',
})
th_usr_set.append({
        0: '0.99',
        1: '0.99',
        2: '0.99',
})

label_name = {
    'ADR': data_args.col_output_adr,
    'pregnancy': data_args.col_output_pregnancy,
    'SS': data_args.col_output_ss,
}
ad_colors = {'ADR': '#e74c3c',
              'pregnancy': '#f39c12',
              'SS': '#3498db'}
pred_scores = np.array([0.0] * 3)

logger.info('Set up Diskcache')
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        background_callback_manager=background_callback_manager,
        prevent_initial_callbacks=True,
)

# navibarのタブ設定
style_button_in_tab = {
    'height': '3.5em',
    'width': '8em',
    'text-align': 'center',
}
tab1_content = dbc.Card(
    dbc.CardBody(
        dbc.Row([
            dbc.Button(
                id='button_predict', n_clicks=0, children='Predict Labels',
                outline=False, class_name='btn btn-primary btn-sm', 
                disabled=True, style=style_button_in_tab,
            ),
            dcc.ConfirmDialog(
                id='confirm_reflect_predictions_to_inputs',
                message='予測結果で全入力を上書きしても良いですか？',
            ),
            dbc.Button(
                id='button_reflect', n_clicks=0, children='All Predictions >> Inputs',
                outline=False, color='btn btn-secondary btn-sm',
                disabled=True, style=style_button_in_tab,
            ),
        ], justify='evenly'),
    )
)
tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.Div([
                dcc.Dropdown(options={'予測': 'フィルター：予測結果', '入力': 'フィルター：入力内容'}, clearable=True,
                                     placeholder='フィルター：なし', disabled=False,
                                     id='filter_target', className='nav-item dropdown'),
                dcc.Dropdown(options={}, clearable=False,
                                     placeholder='対象項目：なし', disabled=False,
                                     id='filter_item', className='nav-item dropdown'),
            ])
        ]
    ),
)
tab3_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row([
                dbc.Button(
                    id='button_th_def', n_clicks=0, children='Default Set',
                    outline=True, class_name='btn btn-primary btn-sm', 
                    disabled=False, style=style_button_in_tab,
                    ),
                dbc.Button(
                    id='button_th_user1', n_clicks=0, children='User Set 1',
                    outline=True, class_name='btn btn-secondary btn-sm', 
                    disabled=False, style=style_button_in_tab,
                    ),
                dbc.Button(
                    id='button_th_user2', n_clicks=0, children='User Set 2',
                    outline=True, class_name='btn btn-light btn-sm', 
                    disabled=False, style=style_button_in_tab,
                    ),
            ], justify='evenly'),
            dbc.Row([
                dbc.Col([
                    html.Div('副作用'),
                ], width=3),
                dbc.Col([
                    dcc.Dropdown(options=th_dropdown_menu_dict[0],
                                 value='youdenj', clearable=False,
                                 disabled=False, id='threshold_adr', className='nav-item dropdown'),
                ]),
            ], align='center'),
            dbc.Row([
                dbc.Col([
                    html.Div('妊婦/授乳'),
                ], width=3),
                dbc.Col([
                    dcc.Dropdown(options=th_dropdown_menu_dict[1],
                                 value='youdenj', clearable=False,
                                 disabled=False, id='threshold_pregnancy', className='nav-item dropdown'),
                ]),
            ], align='center'),
            dbc.Row([
                dbc.Col([
                    html.Div('SS'),
                ], width=3),
                dbc.Col([
                    dcc.Dropdown(options=th_dropdown_menu_dict[2],
                                 value='youdenj', clearable=False,
                                 disabled=False, id='threshold_ss', className='nav-item dropdown'),
                ]),
            ], align='center'),
        ]
    ),
)
tab4_content = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
            dbc.Button(
                id='button_download', n_clicks=0, children='Download',
                outline=False, color='btn btn-secondary btn-sm',
                disabled=True, style=style_button_in_tab,
            ),
            dcc.Download(id='download_dataframe'),
            ], style={'text-align': 'center'}),
            dbc.Col([
            dcc.RadioItems(options=['.xlsx', '.csv/UTF-8', '.csv/Shift-JIS'],
                           value=app_setting['download_default_type'],
                           id='download_radioitems_filetype'),
            ]),
        ],),
        dbc.Row([
            dbc.Col([
                dcc.RadioItems(options=['All Cases　', 'Filtered Cases'], value='All Cases　', id='download_radioitems',),
            ]),
        ]),
    ]),
)

tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label='予測', tab_id='tab-1', class_name='nv-link'),
                dbc.Tab(tab2_content, label='表示', tab_id='tab-2', class_name='nv-link', disabled=app_setting['tab_disabled']['filter']),
                dbc.Tab(tab3_content, label='閾値', tab_id='tab-3', class_name='nv-link', disabled=app_setting['tab_disabled']['threshold']),
                dbc.Tab(tab4_content, label='出力', tab_id='tab-4', class_name='nv-link'),
            ],
            id='tabs',
            active_tab='tab-1',
        ),
    ], className='bg-light', style={'height': '150px', },
)

# top; navibar
navibar = [
        dbc.Col(
            [
            dbc.Row([
                dbc.Col(
                    dcc.Upload(
                        id='upload_csv',
                        children=html.Div([
                            'Drag and Drop or ', html.A('Select File',
                                    className='link'),
                                    ]),
                        className='navbar-light bg-light',
                        style={
                            'min-height': '3.5em',
                            'max-width': '10em',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'text-align': 'center',
                            },
                        # Allow multiple files to be uploaded
                        multiple=False,
                    ),
                ),
            ], align='center'),
            dbc.Row(
                dbc.Col(
                    [
                        html.P(
                            children='No csv is identified.',
                            id='csv_file_name',
                            className='text-white',
                        ),
                        html.P(
                            children='system message.',
                            id='text_display_setting',
                            className='text-white',
                        ),
                    ],
                ),
            ),
            dbc.Row(
                dbc.Col(
                    
                ),
            ),
            ],  width=3, 
        ),
        dbc.Col(
            tabs, width='auto',
        ),
]
progress_bar = dbc.Col(
            [
                dbc.Progress(value=100, color='primary', id="progress_inference",
                            style={'height': '10px', 'margin': '0px 0px'},
                            animated=False, striped=False)
            ]
)

s_viz_stacks = []
for i, key in enumerate(label_name.keys()):
    s_viz_stacks.append(
        dbc.Row(
            dbc.Col(
                html.H6(label_name[key],
                        id=f'item_name_{key}'),
                style={'text-align': 'center',
                       'font-weight': 'bold'}
            ),
        )
    )
    s_viz_stacks.append(
        dbc.Row([
            dbc.Col(
                daq.Gauge(
                color={'gradient':True,
                        'ranges': {'white': [-1, 1],
                                    ad_colors[key]: [1, 1]}},
                value=pred_scores[i],

                id=f'gauge_{key}',
                scale={'interval': 0.1,
                        'labelInterval': 5},
                min=-1, max=1, size=120,
            ), width='auto'),
            dbc.Col(
                [daq.BooleanSwitch(
                on=True,
                id=f'switch_{key}',
                color=ad_colors[key],
                label='Input',
                labelPosition='top',
                ),
                html.Div('予測: - / 入力: - ',
                 id=f'diff_inf_and_input_{key}',
                 style={'text-align': 'center'}),
                ],
                align='center',
            )
        ])
    )
score_viz = dbc.Row(
                dbc.Col(
                    s_viz_stacks
                )
)

move_buttons = [
                dbc.Button('<<', outline=False, color='light', class_name='me-1',
                        id='button_move_to_previous', disabled=True,
                            style={'margin-right': '20px'}),
                dbc.Popover('Previous', target='button_move_to_previous',
                            id='button_move_to_previous_popover',
                            body=True, trigger='hover', autohide=False, placement='top'),
                dbc.Button('>>', outline=False, color='light', class_name='me-1',
                        id='button_move_to_next', disabled=True,
                            style={'margin-left': '20px'}),
                dbc.Popover('Next', target='button_move_to_next',
                            id='button_move_to_next_popover',
                            body=True, trigger='hover', autohide=False, placement='top'),
]

content_title = html.Div([
                html.H6(children='問合せ日時; {}, 問合せID; {}, 製品詳細名; {}.'.format(' - ', ' - ', ' - '),
                        id='inquiry_date_id_drug'
                ),
                dcc.Dropdown(options={0: '問合せ：タイトル'}, value=0, clearable=False,
                                     id='inquiry_title', className='nav-item dropdown'),
])
content_header = dbc.Row([
                    dbc.Col(children=move_buttons, width='auto', align='center'),
                    dbc.Col(children=content_title, width='auto', style={'text-align': 'start'}),
                    ],
                    justify='start',
)

content_right = dbc.Row(
                dbc.Col([
                    dbc.Row(dbc.Col(
                        html.Div([
                            html.H6('問合せ:問題詳細'),
                            html.Iframe(srcDoc='',
                                        id='inquiry_text',
                                        style={'height': '35vh',
                                                'width': '100%'},
                                    ),
                        ]),
                    )),
                    dbc.Row(dbc.Col(
                        html.Div([
                            html.H6('問合せ:回答詳細'),
                            html.Iframe(srcDoc='',
                                        id='response_text',
                                        style={'height': '35vh',
                                                'width': '100%'},
                                    ),
                        ]),
                    ))],
                ),
)
content_left = dbc.Row(
                dbc.Col(children=score_viz,
                )
)

content = dbc.Row(
    dbc.Col([
            dbc.Row(
                [dbc.Col(content_left, width=4),
                 dbc.Col(content_right, width=8)]),
        ]
    ),
)

# Layout
app.layout = dbc.Container(
    [
        html.Div([
            dcc.Store(id='raw_data', storage_type='memory'),
            dcc.Store(id='display_data', storage_type='memory'),
            dcc.Store(id='display_options', storage_type='memory'),
            dcc.Store(id='temporary_inputs', storage_type='memory'),
            dcc.Store(id='scores_predicted', storage_type='memory'),
            dcc.Interval(id='interval_wait', interval=500, disabled=True),
        ]),
        dbc.Row(progress_bar, className='bg-dark'),
        dbc.Row(navibar, className='navbar navbar-expand-lg navbar-dark bg-dark', justify='start'),
        dbc.Row(content_header, className='navbar-light bg-light'),
        dbc.Row(content),
    ],
    fluid=True,
)


filtername_display = {
    'p-ADR': '予測「副作用＋」',
    'p-pregnancy': '予測「妊婦・授乳婦服薬＋」',
    'p-SS': '予測「SS＋」',
    'p-allnegative': '予測「全て”－”」',
    'i-ADR': '入力「副作用＋」',
    'i-pregnancy': '入力「妊婦・授乳婦服薬＋」',
    'i-SS': '入力「SS＋」',
    'no_filtered': '「全て表示」',
    None: 'no filter'
}

# 閾値の設定、選択
@app.callback(
            output=[
            Output('threshold_adr', 'value'),
              Output('threshold_pregnancy', 'value'),
              Output('threshold_ss', 'value'),
            ],
            inputs=dict(
                drop_down_adr=Input('threshold_adr', 'value'),
                drop_down_pregnancy=Input('threshold_pregnancy', 'value'),
                drop_down_ss=Input('threshold_ss', 'value'),
                button_def=Input('button_th_def', 'n_clicks'),
                button_user1=Input('button_th_user1', 'n_clicks'),
                button_user2=Input('button_th_user2', 'n_clicks'),
            ),
)
def set_thresholds(drop_down_adr, drop_down_pregnancy, drop_down_ss,
                   button_def, button_user1, button_user2):
    triggered_id = ctx.triggered_id
    ret_dict = {
        0: drop_down_adr,
        1: drop_down_pregnancy,
        2: drop_down_ss,
    }
    if triggered_id == 'button_th_def':
        ret_dict.update({
            i: 'youdenj' for i in range(3)
        })
    if triggered_id == 'button_th_user1':
        ret_dict.update({
            i: th_usr_set[0][i] for i in range(3)
        })
    if triggered_id == 'button_th_user2':
        ret_dict.update({
            i: th_usr_set[1][i] for i in range(3)
        })
    return [ret_dict[i] for i in range(3)]

# 予測結果で入力を上書きする前に注意喚起
@app.callback(Output('confirm_reflect_predictions_to_inputs', 'displayed'),
              Input('button_reflect', 'n_clicks'),
)
def display_confirm1(value):
    if value:
        return True
    return False

# 表示フィルターの管理
@app.callback(Output('filter_item', 'options'),
              Input('filter_target', 'value'),
)
def set_filter(filter_target_value):
    triggered_id = ctx.triggered_id
    item_dict = {
        '予測': {'p-ADR': '副作用',
                 'p-pregnancy': '妊婦・授乳婦服薬',
                 'p-SS': 'SS',
                 'p-allnegative': '全てnegativeと予測',
                 'no_filtered': '全て表示',
                 },
        '入力': {'i-ADR': '副作用',
                 'i-pregnancy': '妊婦・授乳婦服薬',
                 'i-SS': 'SS',
                 'no_filtered': '全て表示',
                 },
        None: {},
    }
    if triggered_id == 'filter_target':
        return item_dict[filter_target_value]

# ダウンロード
@app.callback(
        output=Output('download_dataframe', 'data'),
        inputs=dict(
            n_click=Input('button_download', 'n_clicks'),
            raw_data=State('raw_data', 'data'),
            display_data=State('display_data', 'data'),
            temporary_inputs=State('temporary_inputs', 'data'),
            message_string=State('csv_file_name', 'children'),
            download_option=State('download_radioitems', 'value'),
            file_type=State('download_radioitems_filetype', 'value'),
        ),
)
def download_file(n_click, raw_data, display_data,
                  temporary_inputs, message_string,
                  download_option, file_type,
                  ):
    logger.info('Download the annotated data.')
    df = pd.DataFrame(raw_data)
    df[data_args.col_output_adr] = temporary_inputs['ADR']
    df[data_args.col_output_pregnancy] = temporary_inputs['pregnancy']
    df[data_args.col_output_ss] = temporary_inputs['SS']

    if download_option == 'All Cases　':
        output = df
    elif download_option == 'Filtered Cases':
        output = df[df.index.isin(display_data['index_list'])]
    else:
        raise ValueError(download_option)

    file_name = re.sub('^.+\"(.+)\"', '\\1', message_string)
    name_stem = re.sub('^(.+)\.csv', '\\1', file_name, flags=re.IGNORECASE)

    if file_type == '.xlsx':
        return dcc.send_data_frame(output.to_excel, filename=name_stem + '.xlsx',
                                index=False, header=True)
    elif file_type.startswith('.csv'):
        if file_type.endswith('UTF-8'):
            out_enc = 'utf-8'
        elif file_type.endswith('Shift-JIS'):
            out_enc = 'cp932'
        csv = output.to_csv(index=False, header=True).encode(out_enc)

        return dcc.send_bytes(src=csv, filename=name_stem + '.csv', 
                    type=f'text/plain; charset="{out_enc}"')


# アップロードファイル読み込みと予測スコアのリセット、予測スコアの入手
@app.callback(
        output=dict(
            raw_data_data=Output('raw_data', 'data'),
            csv_file_name_children=Output('csv_file_name', 'children'),
            scores_predicted_data=Output('scores_predicted', 'data'),
            button_predict_children=Output('button_predict', 'children'),
            button_predict_disabled=Output('button_predict', 'disabled'),
            interval_wait_disabled=Output('interval_wait', 'disabled'),
        ),
        inputs=dict(
            csv_file_name=State('csv_file_name', 'children'),
            list_of_contents=Input('upload_csv', 'contents'),
            list_of_names=State('upload_csv', 'filename'),
            button_predict_n_clicks=Input('button_predict', 'n_clicks'),
            raw_data=State('raw_data', 'data'),
        ),
        background=True,
        running=[
            (Output('button_predict', 'disabled'), True, True),
            (Output('button_predict', 'children'), [dbc.Spinner(size="sm"), " Running.."], 'Finished'),
        ],
        progress=[
            Output("progress_inference", "value"),
            Output("progress_inference", "max"),
        ],
)
def load_file(set_progress,
              csv_file_name, list_of_contents, list_of_names,
              button_predict_n_clicks, raw_data,
):
    triggered_id = ctx.triggered_id
    output = {
        'raw_data_data': raw_data,
        'csv_file_name_children': csv_file_name,
        'scores_predicted_data': None,
        
    }
    def return_load_text(name, n):
        return f'{n} records in "{name}"'

    if triggered_id == 'upload_csv':
        logger.info(f'Upload a csv file{csv_file_name}')
        
        uploaded_data = []
        if list_of_contents is not None:
            c = list_of_contents
            n = list_of_names
            if n.lower().endswith('.csv'):
                uploaded_data.append((c, n))
                _, content_string = c.split(',')
                decoded = base64.b64decode(content_string)
                enc = detect(decoded)
                # print(enc)
                
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode(enc['encoding'])))
                except:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                # config.ymlで設定した列名が存在するかどうかを確認する
                # 存在しない場合は、入力値がblankの列を強制的に作成して表示エラーを回避する
                df = validate_csv_after_load(df, data_args)

                output.update({
                    'raw_data_data': {k: df[k].to_list() for k in df.columns},
                    'csv_file_name_children': return_load_text(uploaded_data[0][1], len(df)),
                    'button_predict_children': 'Predict Labels',
                    'button_predict_disabled': False,
                })
                set_progress((0, 100))
    elif triggered_id == 'button_predict':
        logger.info('PUSH PREDICT BUTTON.')
        df = pd.DataFrame(raw_data)
        logger.info('call an inference model.')
        df = ml_text_classifier(df=df, args=args,
                                progress_function=set_progress, logger=logger)
        logger.info('Success: Got Predict Scores.')

        scores = {f'pred_{col}': df[f'pred_{col}'].to_list()
                         for col in ['ADR', 'pregnancy', 'SS', 'allnegative']}

        output.update({
            'scores_predicted_data': scores,
            'button_predict_children': 'Predicted..',
            'button_predict_disabled': True,
        })
    logger.info('KEYS that is not None: {}'.format(', '.join([key for key in output.keys() if output[key]])))
    return output


@app.callback(
        output=dict(
            display_data=Output('display_data', 'data'),
            temporary_inputs=Output('temporary_inputs', 'data'),
            text_display_setting=Output('text_display_setting', 'children'),
            inquiry_title_options=Output('inquiry_title', 'options'),
            inquiry_title_value=Output('inquiry_title', 'value'),
            inquiry_title_disabled=Output('inquiry_title', 'disabled'),
            inquiry_date_id_drug_children=Output('inquiry_date_id_drug', 'children'),
            inquiry_text_srcDoc=Output('inquiry_text', 'srcDoc'),
            response_text_srcDoc=Output('response_text', 'srcDoc'),
            switch_ADR_on=Output('switch_ADR', 'on'),
            switch_pregnancy_on=Output('switch_pregnancy', 'on'),
            switch_SS_on=Output('switch_SS', 'on'),
            diff_inf_and_input_ADR=Output('diff_inf_and_input_ADR', 'children'),
            diff_inf_and_input_pregnancy=Output('diff_inf_and_input_pregnancy', 'children'),
            diff_inf_and_input_SS=Output('diff_inf_and_input_SS', 'children'),
            gauge_ADR_value=Output('gauge_ADR', 'value'),
            gauge_pregnancy_value=Output('gauge_pregnancy', 'value'),
            gauge_SS_value=Output('gauge_SS', 'value'),
            button_reflect_disabled=Output('button_reflect', 'disabled'),
            button_move_to_previous_disabled=Output('button_move_to_previous', 'disabled'),
            button_move_to_next_disabled=Output('button_move_to_next', 'disabled'),
            button_move_to_previous_popover_children=Output('button_move_to_previous_popover', 'children'),
            button_move_to_next_popover_children=Output('button_move_to_next_popover', 'children'),
            button_download_disabled=Output('button_download', 'disabled'),
        ),
        inputs=dict(
            raw_data=Input('raw_data', 'data'),
            button_move_to_previous=Input('button_move_to_previous', 'n_clicks'),
            button_move_to_next=Input('button_move_to_next', 'n_clicks'),
            inquiry_title_value=Input('inquiry_title', 'value'),
            switch_adr_on=Input('switch_ADR', 'on'),
            switch_pregnancy_on=Input('switch_pregnancy', 'on'),
            switch_ss_on=Input('switch_SS', 'on'),
            button_reflect_confirm=Input('confirm_reflect_predictions_to_inputs', 'submit_n_clicks'),
            filter_item_value=Input('filter_item', 'value'),
            threshold_adr_value=Input('threshold_adr', 'value'),
            threshold_pregnancy_value=Input('threshold_pregnancy', 'value'),
            threshold_ss_value=Input('threshold_ss', 'value'),
        ),
        state=dict(
            scores_predicted=State('scores_predicted', 'data'),
            display_data_now=State('display_data', 'data'),
            temporary_inputs=State('temporary_inputs', 'data'),
        ),
)
def extract_data_to_show(raw_data, scores_predicted,
                         button_move_to_previous, button_move_to_next,
                         inquiry_title_value,
                         switch_adr_on, switch_pregnancy_on, switch_ss_on,
                         button_reflect_confirm,
                         filter_item_value,
                         threshold_adr_value, threshold_pregnancy_value, threshold_ss_value,
                         display_data_now, temporary_inputs,):
    def refresh_display(output, temporary_inputs):
        output['display_data'] = output.copy()
        output['temporary_inputs'] = temporary_inputs
        del output['index_list'], output['display_index']
        return output
    def move_to(triggered_id, df_index):
        def get_prev_next_index(disp_index):
            n_record = len(df_index)
            i_prev = df_index.index(df_index[disp_index]) - 1
            i_next = disp_index + 1 if disp_index != n_record - 1 else 0
            return df_index[i_prev], df_index[i_next]
        def text_move_hover(i):
            string = raw_data[data_args.col_title][i]
            return f'{i + 1}. {string}'
        
        if index in df_index:
            idx_in_df_index = df_index.index(index)
        else:
            idx_in_df_index = 0

        # disp_index: df_index内のindex。df_index配列の各valueではない。
        if triggered_id == 'button_move_to_previous':
            disp_index = idx_in_df_index - 1 if idx_in_df_index != 0 else idx_in_df_index - 1
        elif triggered_id == 'button_move_to_next':
            disp_index = idx_in_df_index + 1 if idx_in_df_index != len(df_index) - 1 else 0
        elif triggered_id == 'inquiry_title':
            if inquiry_title_value:
                disp_index = df_index.index(int(inquiry_title_value))
            else:
                disp_index = 0
        elif triggered_id == 'filter_item':
            if len(df_index):
                if index in df_index:
                    disp_index = idx_in_df_index
                else:
                    disp_index = 0
            else:
                return disp_data_def
        else:
            disp_index = df_index.index(index)

        i_prev, i_next = get_prev_next_index(disp_index)

        return {
            'display_index': df_index[disp_index],
            'index_list': df_index,
            'button_move_to_previous_popover_children': text_move_hover(i_prev), 
            'button_move_to_next_popover_children': text_move_hover(i_next),
        }
    def create_index_list(data_length, filter_info):
        all_index = np.arange(data_length)
        if not filter_info or filter_info == 'no_filtered':
            index_list = all_index
        elif filter_info.startswith('p-'):
            if not scores_predicted:
                return all_index.tolist()
            f_name = filter_info.replace('p-', '')
            np_scores = {key: np.array(scores_predicted[f'pred_{key}']) for key in score_th.keys()}
            try:
                if f_name != 'allnegative':
                    preds = np_scores[f'{f_name}'] >= score_th[f_name]
                else:
                    preds = (np_scores[f'ADR'] < score_th['ADR']) & \
                            (np_scores[f'pregnancy'] < score_th['pregnancy']) & \
                            (np_scores[f'SS'] < score_th['SS'])
                index_list = all_index[preds]    
            except:
                index_list = []
        elif filter_info.startswith('i-'):
            f_name = filter_info.replace('i-', '')
            np_inputs = {key: np.array(temporary_inputs[key]) for key in score_th.keys()}
            try:
                inputs = np.array(np_inputs[f_name]) == 1
                index_list = all_index[inputs]
            except:
                index_list = []
        else:
            raise ValueError(filter_info)
        return list(index_list)
    def get_filtered_info(index_list, filter_item_value):
        return f'{len(index_list)} records filtered by "{filtername_display[filter_item_value]}."'
    
    output = dict()
    triggered_id = ctx.triggered_id
    if app_setting['debug_mode']:
        logger.info(f'Triggered_prop IDs in extract_data_to_show: {ctx.triggered_prop_ids}')

    switch_value = {
        'ADR': switch_adr_on,
        'pregnancy': switch_pregnancy_on,
        'SS': switch_ss_on,
    }
    score_th = {'ADR': th_dict[0][threshold_adr_value],
            'pregnancy':  th_dict[1][threshold_pregnancy_value],
            'SS':  th_dict[2][threshold_ss_value],
    }
    if not raw_data:
        output.update(disp_data_def)
        return refresh_display(output, temporary_inputs)

    # ファイルアップロード直後の表示反映
    if (triggered_id == 'raw_data' and not scores_predicted) or display_data_now['display_index'] == -1:
        index = 0
        output.update(disp_data_def)
        output['display_index'] = index
        temporary_inputs = {
                    'ADR': raw_data[data_args.col_output_adr],
                    'pregnancy': raw_data[data_args.col_output_pregnancy],
                    'SS': raw_data[data_args.col_output_ss],
        }

        index_list = create_index_list(data_length=len(raw_data[data_args.col_id]),
                                        filter_info=filter_item_value)
        dict_case_title = {str(i): f'{i + 1}. {raw_data[data_args.col_title][i]}' for i in index_list}
    else:
        index = display_data_now['display_index']
        index_list = display_data_now['index_list']
        dict_case_title = display_data_now['inquiry_title_options']

    output.update({
        'button_move_to_previous_disabled': False,
        'button_move_to_next_disabled': False,
        'button_download_disabled': False,
    })


    # フィルターを設定したときの表示内容変更（index_listの更新と表示更新）
    if triggered_id in {'filter_item', 'threshold_adr', 'threshold_pregnancy', 'threshold_ss'}:
        index_list = create_index_list(data_length=len(raw_data[data_args.col_id]),
                                         filter_info=filter_item_value)
        if len(index_list):
            # if index not in index_list:
            #     index = index_list[0]
            output['display_index'] = index

    # フィルター設置状態でファイルを読み込んだときの対応
    # 表示事例数が0の場合は、move_toを呼び出す前に値を返す
    if not len(index_list):
        output.update(disp_data_def)
        output['text_display_setting'] = get_filtered_info(index_list, filter_item_value)
        return refresh_display(output, temporary_inputs)

    # Predict Labelsを押したときの現在画面保持
    # フィルターによって非表示にされる場合は、最初の事例を表示
    if triggered_id == 'raw_data' and scores_predicted:
        index_list = create_index_list(data_length=len(raw_data[data_args.col_id]),
                                         filter_info=filter_item_value)
        if index not in index_list:
            index = index_list[0]    

    output.update(move_to(triggered_id, index_list))
    index = output['display_index']

    dict_case_title = {str(i): f'{i + 1}. {raw_data[data_args.col_title][i]}' for i in index_list}
    
    # 画面に表示される事例の詳細
    output.update({
        'inquiry_title_options': dict_case_title,
        'inquiry_title_disabled': False,
        'inquiry_title_value': str(index),
        'inquiry_date_id_drug_children': '問合せ日時; {}, 問合せID; {}, 製品詳細名; {}.'.format(
                                            raw_data[data_args.col_date][index],
                                            raw_data[data_args.col_id][index],
                                            raw_data[data_args.col_drugname][index]),
        'inquiry_text_srcDoc': raw_data[data_args.sentence1_key][index],
        'response_text_srcDoc': raw_data[data_args.sentence2_key][index],
    })

    # 予測スコアの有無でボタンの有効/無効判定
    if not scores_predicted:
        output.update({
            'button_reflect_disabled': True,
        })
        output.update({
            f'gauge_{item}_value': 0 for item in score_th.keys()
        })
        pred_info = {item: '-' for item in score_th.keys()}
    else:
        output.update({
            f'gauge_{item}_value': get_modified_value(value=scores_predicted[f'pred_{item}'][index],
                                          threshold=th) for item, th in score_th.items()
        })
        pred_info = {item: 'あり' if scores_predicted[f'pred_{item}'][index] >= th else 'なし'
                                                        for item, th in score_th.items()}
        output.update({
            'button_reflect_disabled': False,
        })

    # スイッチ入力をtemporary_inputsに反映
    for item in score_th.keys():
        if triggered_id == f'switch_{item}':
            temporary_inputs[item][index] = int(switch_value[item])
    
    # 予測をInputに反映させるボタンを押した場合
    if triggered_id == 'confirm_reflect_predictions_to_inputs':
        temporary_inputs = {item: (np.array(scores_predicted[f'pred_{item}']) >= th).astype(int).tolist()
                                                        for item, th in score_th.items()}
    
    #入力情報の有無（アップロード前のCSV含む）でスイッチ状態に反映
    try:
        switch_state = {f'switch_{item}_on': temporary_inputs[item][index]
                             for item in score_th.keys()}
        switch_info = {item: 'あり' if switch_state[f'switch_{item}_on'] else 'なし'
                                                         for item in score_th.keys()}
    except:
        switch_state = {f'switch_{item}_on': disp_data_def[f'switch_{item}_on']
                             for item in score_th.keys()}
        switch_info = {item: '-' for item in score_th.keys()}
    
    output.update(switch_state)
    output.update({
        f'diff_inf_and_input_{item}': '予測: {} / 入力: {}'.format(pred_info[item], switch_info[item])
                                        for item in score_th.keys()
    })

    # フィルター情報を含めて画面に表示
    output['text_display_setting'] = get_filtered_info(index_list, filter_item_value)
    
    return refresh_display(output, temporary_inputs)
        
if __name__ == "__main__":
    logger.info('Run App.')
    app.run(
        debug=app_setting['debug_mode'],
        port=app_setting['port'],
    )
    