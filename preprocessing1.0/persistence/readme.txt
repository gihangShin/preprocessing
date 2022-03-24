Postgre SQL

model단
model링 필요할지 의문

test 방법
    httpie
        - httpie 는 매번 호출 시 session 초기화 -> session 등록, 사용해야함

        - '/' session에 df 등록 테스트 코드 작성 해둠
            http --session=gihang -v POST localhost:5000/

        - df 결측치 처리 동작 수행
            http --session=gihang -v POST localhost:5000/preprocessing/missingvalue m_value=remove columns=jibun
            http --session=gihang -v POST localhost:5000/preprocessing/missingvalue m_value=ffill columns=exclusive_use_area

        - 수행 후 dataset(db)테이블에 수행 결과 등록 결과 예시
            admin155	1.01	admin	1	request_user	testname	{"function": "missing_data", "selected_column": "exclusive_use_area"}		2022-03-24 14:59:22.476
            admin139    1.02	admin	2	request_user	testname	{"function": "missing_data", "selected_column": "exclusive_use_area"}		2022-03-24 14:59:28.618
            admin265	1.03	admin	3	request_user	testname	{"function": "missing_data", "selected_column": "jibun"}		2022-03-24 14:59:36.120