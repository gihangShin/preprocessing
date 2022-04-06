import json
import math
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from flask import session
import dataset
import handling_dataset


class Preprocessing:

    def __init__(self, app, dsDAO, jhDAO):
        self.app = app
        self.dsDAO = dsDAO
        self.jhDAO = jhDAO
        self.hd = handling_dataset.HandlingDataset(app, dsDAO, jhDAO)

    #######################################################################################
    #######################################################################################
    #######################################################################################

    # (처음) 불러오기
    def load(self, payload):
        ds = dataset.Dataset(payload)
        ds.load_dataset_from_warehouse_server()
        ds = self.hd.sampling_dataset(ds)

        return ds.dataset_and_dtypes_to_json()

    # 가공 처리 동작
    def preprocessing(self, payload, method):
        ds = dataset.Dataset(payload)
        ds.load_dataset_from_request(payload)
        ds.set_job_id(method)

        if method == 'delete_column':
            ds = self.hd.delete_column(ds)
        elif method == 'missing_value':
            ds = self.hd.missing_value(ds)
        elif method == 'set_col_prop':
            ds = self.hd.set_col_prop(ds)
        elif method == 'set_col_prop_to_datetime':
            ds = self.hd.set_col_prop_to_datetime(ds)
        elif method == 'split_datetime':
            ds = self.hd.split_datetime(ds)
        elif method == 'dt_to_str_format':
            ds = self.hd.dt_to_str_format(ds)
        elif method == 'diff_datetime':
            ds = self.hd.diff_datetime(ds)
        elif method == 'change_column_order':
            ds = self.hd.change_column_order(ds)
        elif method == 'case_sensitive':
            ds = self.hd.case_sensitive(ds)
        elif method == 'replace_by_input_value':
            ds = self.hd.replace_by_input_value(ds)
        elif method == 'remove_space_front_and_rear':
            ds = self.hd.remove_space_front_and_rear(ds)
        elif method == 'drop_duplicate_row':
            ds = self.hd.drop_duplicate_row(ds)
        elif method == 'calculating_column':
            ds = self.hd.calculating_column(ds)
        else:
            print('ERRORERRORERRORERRORERROR')

        if method == 'show_duplicate_row':
            return json.dumps({
                'dataset': ds.dataset_to_json(),
                'dataset_dtypes': ds.get_types(),
                'duplicate_values': self.hd.show_duplicate_row(ds)
            }, ensure_ascii=False)

        return ds.dataset_and_dtypes_to_json()

    # 데이터셋 추출
    def export(self, payload):
        ds = dataset.Dataset(payload)
        ds.load_dataset_from_warehouse_server()

        # redo
        ds = self.hd.redo_job_history(ds=ds)
        ds.export()
        return "message"
