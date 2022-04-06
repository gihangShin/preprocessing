import json

import pandas as pd


class Dataset:

    def __int__(self):
        pass

    def __init__(self, params):
        self.project_id = params['project_id']
        self.file_id = params['file_id']
        self.version = float(params['version'])
        self.dataset = None
        self.data_types = None
        if 'params' in params:
            self.job_params = params['params']
        self.job_id = None

    ###############################################
    # init
    def load_dataset_from_warehouse_server(self):
        # 테스트용 코드
        project_dir = './server/%s/p_data/' % self.project_id
        full_file_name = '%s_V%.2f.json' % (self.file_id, self.version)

        url = project_dir + full_file_name
        print("######################")
        print(url)

        self.dataset = pd.read_json(url)
        self.dataset.convert_dtypes()
        self.data_types = self.get_types()

        ##################
        # 개발 서버에서 실제 코드
        # 데이터를 보관한 서버에서 데이터셋을 가져와야함

        return self

    def load_dataset_from_request(self, payload):
        # json에서 load
        # dataset_type도 객체 내 저장
        self.dataset = pd.read_json(payload['dataset'])
        self.data_types = payload['dataset_dtypes']
        self.sync_dataset_with_dtypes()
        return self

    ###############################################

    def export_dataset(self):
        # 추출
        url = "./server/%s/p_data/%s_V%.2f.json" % (self.project_id, self.file_id, self.version)
        self.dataset.to_json(url, force_ascii=False)

    def load_dataset(self, params):
        # dataset 불러오기
        # project_id, file_id, version

        pass

    def get_types(self):
        data_types = self.dataset.dtypes.to_dict()
        self.data_types = dict()
        for k, v in data_types.items():
            self.data_types[k] = str(v)
        return self.data_types

    def sync_dataset_with_dtypes(self):
        self.dataset = self.dataset.astype(self.data_types)
        return self

    def dataset_to_json(self):
        return self.dataset.to_json(force_ascii=False)

    def job_params_to_json(self):
        return json.dumps(self.job_params, ensure_ascii=False)

    def dataset_and_dtypes_to_json(self):
        self.dataset = self.dataset.astype(str)
        return json.dumps({
            'dataset': self.dataset_to_json(),
            'dataset_dtypes': self.data_types
        }, ensure_ascii=False)

    def set_job_params(self, job_params):
        self.job_params = job_params
        return self

    def set_job_id(self, job_id):
        self.job_id = job_id
        return self
