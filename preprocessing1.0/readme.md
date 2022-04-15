
## Request parameter
 - default parameter
   - file_id, project_id, version
 - load
   - file_id, project_id, version
 - preprocessing
   - default
   - dataset, dataset_dtypes
   - params = {'column_id',...}
 - export
   - 추출에 필요한 정보
     - default
   - 원본 파일에 접근하여 job_history의 동작 재수행

## Response parameter
 - default parameter
   - dataset, dataset_dtypes
 - load
   - 원본 파일에서 dataset을 불러온 후, sampling 정보를 통해 sampling된 dataset과 데이터셋의 컬럼 속성 정보 전달
   - default or error message
 - preprocessing
   - 요청 받은 동작 수행 후 DB job_history 내 저장
   - default or error message
 - export
   - 원본 파일을 읽은 후 재수행 그리고 버전이 올라간 상태로 추출
   - 최신화 된 default param, 성공, 실패 message

## how to start flask
- powershell
  - $env:FLASK_APP = "helloworld.py"
  - $env:FLASK_ENV = "development"
  - flask run
- cmd
  - set FLASK_DEBUG=1
  - set FLASK_APP=app
  - flask run

## 현재 기능
- 불러오기
  1.load

- 전처리 동작
  - redirect_preprocess (스위치 기능 수행)
    - prefix url : /preprocessing/...
      1. delete_column
      2. missing_value
      3. set_col_prop
      4. set_col_prop_to_datetime
      5. split_datetime
      6. dt_to_str_format
      7. diff_datetime
      8. change_column_order
      9. case_sensitive
      10. replace_by_input_value
      11. remove_space_front_and_rear
      12. drop_duplicate_row
      13. calculating_column
      14. drop_row
      15. rename_col
      16. split_col
      17. missing_data_model
      18. unit_conversion
      19. concat
      20. merge
  - 조회기능
    - show_duplicate_row
    - show_conditioned_row

- 추출
  - export
    1. load_dataset_from_warehouse_server
    2. redo_job_history
    3. export_dataset