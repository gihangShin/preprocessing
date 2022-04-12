import json
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from flask import session


class Preprocessing:

    def __init__(self, hd, dataset):
        self.hd = hd
        self.dataset = dataset
        # self.hd = handling_dataset.HandlingDataset(app, dsDAO, jhDAO)

    #######################################################################################
    #######################################################################################
    #######################################################################################

    # (처음) 불러오기
    def load(self, payload):
        ds = self.dataset.Dataset(payload)
        ds.load_dataset_from_warehouse_server()
        ds = self.hd.sampling_dataset(ds)

        return ds.dataset_and_dtypes_to_json()

    # 가공 처리 동작
    def preprocessing(self, payload, job_id):
        ds = self.dataset.Dataset(payload)
        ds.load_dataset_from_request(payload)
        ds.set_job_id(job_id)

        # 가공동작 통합 redirect
        ds = self.hd.redirect_preprocess(ds)

        self.hd.insert_job_history_into_database(ds)
        return ds.dataset_and_dtypes_to_json()

    # 조회동작 구분 필요
    def show(self, payload, job_id):
        ds = self.dataset.Dataset(payload)
        ds.load_dataset_from_request(payload)
        ds.set_job_id(job_id)

        result = {
            'dataset': ds.dataset_to_json(),
            'dataset_dtypes': ds.get_types()
        }

        if job_id == 'show_duplicate_row':
            result[job_id] = self.hd.show_duplicate_row(ds).to_dict()
        elif job_id == 'show_conditioned_row':
            result[job_id] = self.hd.show_conditioned_row(ds).to_dict()

        return json.dumps(result, ensure_ascii=False)

    # 데이터셋 추출
    def export(self, payload):
        ds = self.dataset.Dataset(payload)
        ds.load_dataset_from_warehouse_server()

        # redo
        ds = self.hd.redo_job_history(ds=ds)
        print(ds.dataset.head())
        print(ds.dataset.describe())
        print(ds.dataset.shape)
        print(ds.data_types)
        ds.export_dataset()
        return "추출 완료"
