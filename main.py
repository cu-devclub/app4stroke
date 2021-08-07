import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from starlette.concurrency import run_in_threadpool
from utils.clinic_utils import extract_handcraft_features
from utils.cam import overlay_cam
from dicom_loader import get_flip_ct_array_loader
from scipy.special import expit as sigmoid
from fastapi import FastAPI, Request
from PIL import Image
from xgboost import XGBClassifier
import uvicorn
import onnxruntime
import torch
import shap
import pandas as pd
import numpy as np
import json
import time
import base64
import io

with open('/app4stroke_ml/configs.json', 'r') as f:
    config = json.load(f)

if config['device'] == 'auto':
    PROVIDER = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
elif config['device'] == 'cuda':
    PROVIDER = 'CUDAExecutionProvider'
else:
    PROVIDER = 'CPUExecutionProvider'

clinic_model = XGBClassifier()
clinic_model.load_model(config['clinic_model_path'])
clinic_explainer = shap.TreeExplainer(clinic_model)

ct_model = onnxruntime.InferenceSession(
    config['ct_model_path'], providers=[PROVIDER])

batch_size = config['batch_size']

app = FastAPI()


@app.post("/api/analyse_dicom/")
async def analyse_dicom_api(request: Request):
    return await analyse_dicom_async(request)


@app.post("/api/predict_prob/")
async def predict_api(request: Request):
    return await predict_async(request)


async def analyse_dicom_async(request):
    dicom_paths = (await request.json())['dicom_paths']
    return await run_in_threadpool(analyse_dicom, dicom_paths)


def analyse_dicom(dicom_paths):
    
    total_slices = 0
    img_bytes, heatmap_bytes, ct_scores = [], [], []
    max_score_slice = 0
    max_ct_score = -1

    load_time = 0
    inference_time = 0
    overlay_cam_time = 0
    total_time = 0

    res = 'no file'
    if len(dicom_paths) > 0:
        t0 = time.time()
        data_loader = get_flip_ct_array_loader(
            dicom_paths, batch_size=batch_size)
        t1 = time.time()
        total_slices = len(data_loader.dataset)
        ct_scores, heatmaps = predict_ct(ct_model, data_loader)
        t2 = time.time()
        imgs = data_loader.dataset.X[..., 2]
        cams = [overlay_cam(img, heatmap)
                for img, heatmap in zip(imgs, heatmaps)]
        t3 = time.time()

        img_bytes = list(map(encode_img, imgs))
        heatmap_bytes = list(map(encode_img, cams))

        max_ct_score = ct_scores.max().item()
        max_score_slice = np.argmax(ct_scores).item() + 1

        t4 = time.time()

        load_time = t1 - t0
        inference_time = t2 - t1
        overlay_cam_time = t3 - t2
        total_time = t4 - t0

        res = 'complete'

    ct_results = {
        "total_slices": total_slices,
        "max_score_slice": max_score_slice,
        "max_ct_score": max_ct_score,
        "img_bytes": img_bytes,
        "heatmap_bytes": heatmap_bytes,
        "ct_scores": ct_scores.tolist(),
        "load_time": load_time,
        "inference_time": inference_time,
        "overlay_cam_time": overlay_cam_time,
        "total_time": total_time,        
        "result": res,
    }

    return ct_results


def predict_ct(ct_model, data_loader):
    scores = []
    heatmaps = []
    for x in data_loader:
        ort_inputs = {ct_model.get_inputs()[0].name: x.numpy()}
        ort_outs = ct_model.run(None, ort_inputs)
        scores.append(ort_outs[0])
        heatmaps.append(ort_outs[1])
    scores = sigmoid(np.concatenate(scores, 0)[:, 0])
    heatmaps = sigmoid(np.concatenate(heatmaps, 0)[:, 0])
    return scores, heatmaps


def encode_img(image):
    pil_img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    return base64.encodebytes(img_byte_arr.getvalue()).decode('utf-8')


async def predict_async(request):
    clinic_data = await request.json()
    return await run_in_threadpool(predict, clinic_data)


def predict(clinic_data):

    clinic_features = extract_handcraft_features(clinic_data)
    clinic_features['max ct score'] = clinic_data['max_ct_score']

    prob, top_pos_factors, top_pos_values, top_pos_impacts, top_neg_factors, top_neg_values, top_neg_impacts = predict_clinic(
        clinic_explainer, clinic_features)

    clinic_results = {                           
                        'prob': prob,
                        'top_pos_factors': top_pos_factors,
                        'top_pos_values': top_pos_values,
                        'top_pos_impacts': top_pos_impacts,
                        'top_neg_factors': top_neg_factors,
                        'top_neg_values': top_neg_values,
                        'top_neg_impacts': top_neg_impacts,
                    }

    return clinic_results


def predict_clinic(clinic_explainer, clinic_features):
    x = pd.DataFrame(clinic_features, index=['value'])
    shap_values = clinic_explainer.shap_values(x)[0]
    af_prob = float(clinic_model.predict_proba(x)[0, 1])

    x = x.T
    x['impact'] = shap_values
    x = x.sort_values('impact').astype(str)

    top_pos = x.iloc[len(x):len(x)-6:-1]
    top_pos_factors = top_pos.index.to_list()
    top_pos_values = top_pos['value'].to_list()
    top_pos_impacts = top_pos['impact'].to_list()

    top_neg = x.iloc[:5]
    top_neg_factors = top_neg.index.to_list()
    top_neg_values = top_neg['value'].to_list()
    top_neg_impacts = top_neg['impact'].to_list()

    return af_prob, top_pos_factors, top_pos_values, top_pos_impacts, top_neg_factors, top_neg_values, top_neg_impacts


if __name__ == "__main__":
    
    host = config['host']
    if host.startswith('http://'):
        host = host[7:]
    elif host.startswith('https://'):
        host = host[8:]

    uvicorn.run(app, 
                host=host, 
                port=config['port'], 
                ssl_keyfile=config['ssl_keyfile'] if config['ssl_keyfile'] else None, 
                ssl_certfile=config['ssl_certfile'] if config['ssl_certfile'] else None,
                ssl_keyfile_password=config['ssl_keyfile_password'] if config['ssl_keyfile_password'] else None
                )
