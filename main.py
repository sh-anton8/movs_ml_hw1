from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
import pandas as pd
import joblib
import numpy as np

FEATURES_SUBSET_NUM_load = joblib.load('data/model/num_feature_list.pkl')
FEATURES_SUBSET_OBJ_load = joblib.load('data/model/obj_feature_list.pkl')
FEATURES_SUBSET_OTHER_load = joblib.load('data/model/other_feature_list.pkl')
fill_nans_data_load = joblib.load('data/model/fill_nan_dict.pkl')
pipe_loaded = joblib.load('data/model/pipeline.pkl')

def parse_torque(df_table):
    """
    Как бы мы жили без if и else? И на этот риторический вопрос даже не ответишь
    "Если бы да кабы", ведь в ответе содержится if
    """
    max_torque_rpm_arr, torque_only_arr = [], []
    for row_idx, row in df_table.iterrows():
        if str(row['torque']) != 'nan':
            torque_str = str(row['torque']).lower()
            if '@' in torque_str:
                torque_split = torque_str.split('@')
            elif 'at' in torque_str:
                torque_split = torque_str.split('at')
            elif '/' in torque_str:
                torque_split = torque_str.split('/')
            else:
                torque_only = float(torque_str.split('nm')[0].strip())
                max_torque_rpm = None

            torque_only_str = torque_split[0]
            if 'nm' in torque_only_str:
                torque_only = float(torque_str.split('nm')[0].strip())
            elif 'kgm' in torque_only_str:
                torque_only = float(torque_str.split('kgm')[0].strip()) * 9.80665
            else:
                if '(' in torque_only_str:
                    torque_only_str = torque_only_str.split('(')[0].strip()
                torque_only = float(torque_only_str)
                if torque_only < 90:
                    torque_only *= 9.80665

            max_torque_rpm_str = torque_split[1]
            if '-' in max_torque_rpm_str:
                max_torque_rpm_str = max_torque_rpm_str.split('-')[1]
            elif '~' in max_torque_rpm_str:
                max_torque_rpm_str = max_torque_rpm_str.split('~')[1]

            if 'rpm' in max_torque_rpm_str:
                max_torque_rpm_str = max_torque_rpm_str.replace(',', '')
                max_torque_rpm = float(max_torque_rpm_str.split('rpm')[0])
            elif '(' in max_torque_rpm_str:
                max_torque_rpm_str = max_torque_rpm_str.replace(',', '')
                max_torque_rpm = float(max_torque_rpm_str.split('(')[0])
            else:
                max_torque_rpm_str = max_torque_rpm_str.replace(',', '')
                max_torque_rpm = float(max_torque_rpm_str) 
        else:
            max_torque_rpm, torque_only = None, None
        max_torque_rpm_arr.append(max_torque_rpm)
        torque_only_arr.append(torque_only)
    return max_torque_rpm_arr, torque_only_arr

def preprocess_data(df, fill_medians):
    max_torque_rpm_arr, torque_only_arr = parse_torque(df)
    df['max_torque_rpm'] = max_torque_rpm_arr
    df['torque_only'] = torque_only_arr
    
    if 'selling_price' in df.columns.to_list():
        df = df.drop(columns=['selling_price'])
    if 'torque' in df.columns.to_list():
        df = df.drop(columns=['torque'])
    
    df['nan_mileage'] = 0 
    df.loc[df['mileage'].isna(), 'nan_mileage'] = 1
    
    df['nan_engine'] = 0 
    df.loc[df['engine'].isna(), 'nan_engine'] = 1
    
    df['nan_max_power'] = 0 
    df.loc[df['max_power'].isna(), 'nan_max_power'] = 1
    
    df['nan_seats'] = 0 
    df.loc[df['seats'].isna(), 'nan_seats'] = 1
    
    df['nan_max_torque_rpm'] = 0 
    df.loc[df['max_torque_rpm'].isna(), 'nan_max_torque_rpm'] = 1
    
    df['nan_torque_only'] = 0
    df.loc[df['torque_only'].isna(), 'nan_torque_only'] = 1
    
    df['name'] = df['name'].apply(lambda x: x.split()[0].lower())
    
    df['year_old'] = 0
    df.loc[(df['year'] < 1990), 'year_old'] = 1
    df['year_90'] = 0
    df.loc[(df['year'] < 2000) & ((df['year'] >= 1990)), 'year_90'] = 1
    df['year_00'] = 0
    df.loc[(df['year'] < 2010) & ((df['year'] >= 2000)), 'year_00'] = 1
    df['year_10'] = 0
    df.loc[(df['year'] < 2020) & ((df['year'] >= 2010)), 'year_10'] = 1
    df['year_new'] = 0
    df.loc[(df['year'] >= 2020), 'year_new'] = 1
    
    df = df.fillna(value=fill_medians)
    df['mileage'] = df['mileage'].apply(lambda x: float(str(x).split()[0]))
    df['engine'] = df['engine'].apply(lambda x: float(str(x).split()[0]))
    df['max_power'] = df['max_power'].apply(lambda x: str(x).strip())
    df['max_power'] = df['max_power'].apply(lambda x: float(x.split()[0]) if x[0] != 'b' else 0)
    df['seats'] = df['seats'].apply(lambda x: int(x))
    df['engine'] = df['engine'].apply(lambda x: int(x))
    
    # этим также неявно добавили произведения и умножения данных признаков (тк log ab = log a + log b)
    df['log_km_driven'] = df['km_driven'].apply(lambda x: np.log(x + 1))
    df['log_engine'] = df['engine'].apply(lambda x: np.log(x + 1))
    df['log_max_power'] = df['max_power'].apply(lambda x: np.log(x + 1))
    df['log_mileage'] = df['mileage'].apply(lambda x: np.log(x + 1))
    df['year_old_seats'] = df.apply(lambda x: int(x.year_old * x.seats), axis=1)
    df['year_90_seats'] = df.apply(lambda x: int(x.year_90 * x.seats), axis=1)
    df['year_00_seats'] = df.apply(lambda x: int(x.year_00 * x.seats), axis=1)
    df['year_10_seats'] = df.apply(lambda x: int(x.year_10 * x.seats), axis=1)
    df['year_new_seats'] = df.apply(lambda x: int(x.year_new * x.seats), axis=1)
    
    df['0mileage'] = 0
    df.loc[(df['mileage'] == 0.0), '0mileage'] = 1
    df['0max_power'] = 0
    df.loc[(df['max_power'] == 0.0), '0max_power'] = 1
    df['0max_torque_rpm'] = 0
    df.loc[(df['max_torque_rpm'] == 0.0), '0max_torque_rpm'] = 1
    df['0torque_only'] = 0
    df.loc[(df['torque_only'] == 0.0), '0torque_only'] = 1
    return df

app = FastAPI()

app.mount("/server_disk", StaticFiles(directory="server_disk"), name="server_disk")
app.mount("/data", StaticFiles(directory="data"), name="data")
templates = Jinja2Templates(directory="server_disk/templates")

@app.get("/")
async def root(request: Request, message='Main page'):
    # return {"message": "Hello world"}
    return templates.TemplateResponse('index.html',
                                      {"request": request,
                                       "message": message})

@app.post("/upload_one")
async def upload(request: Request,
                 upload_file: UploadFile = File(...)):

    json_data = json.load(upload_file.file)
    print(json_data)   

    for k, v in json_data.items():
        json_data[k] = [v]

    df = pd.DataFrame.from_dict(json_data)
    df = preprocess_data(df, fill_nans_data_load)
    pred = pipe_loaded.predict(df)
    pred_fin = str(np.exp(pred)[0] - 1)

    message='Main page'
    if 'selling_price' not in json_data:
        return templates.TemplateResponse('index_with_ans.html',
                                          {"request": request,
                                           "message": message,
                                           "pred_fin": pred_fin})
    else:
        tgt_fin = str(json_data['selling_price'][0])
        return templates.TemplateResponse('index_with_tgt.html',
                                          {"request": request,
                                           "message": message,
                                           "pred_fin": pred_fin,
                                           "tgt_fin": tgt_fin})

@app.post("/upload_many")
async def upload(request: Request,
                 upload_file: UploadFile = File(...)):

    df = pd.read_csv(upload_file.file)
    print(df.columns.to_list())
    df2 = preprocess_data(df, fill_nans_data_load)
    pred = pipe_loaded.predict(df2)
    pred_fin = np.exp(pred) - 1
    df['predicted_price'] = pred_fin
    df.to_csv("data/preds.csv", index=False)
    return FileResponse("data/preds.csv")
