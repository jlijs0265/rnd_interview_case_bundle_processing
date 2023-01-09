## Bundle Processor

### 소스 파일 구성

* bundle_processor.py : 묶음(Bundle)간 주문(WayPoint)을 재분배하는 알고리즘
* clustering.py : k-means 클러스터링 알고리즘
* resources/sample_bundles.json : 샘플 묶음 데이터
* outputs/bundle_before.png : 묶음 데이터 재조정 전 분포 (예시)
* outputs/bundle_after.png : 묶음 데이터 재조정 후 분포 (예시)

### bundle_processor 주요 구성요소

* Location : 위치 객체
* WayPoint : (당일배송 주문의) 배송지 객체
* Cluster : bundle 내의 waypoints를 지리적 인접도를 기준으로 분할하여 저장하는 객체
* Bundle : (당일배송 주문) 묶음 객체
    * cluster_waypoints : waypoints를 지리적 인접도를 기준으로 분할
    * evaluate_clusters_and_sort : 각 cluster의 지리적 인접도를 기준으로 cost를 부여하고 정렬
    * set_outliers : 지리적 인접도를 기준으로 부여된 cost가 높은 cluster를 분류
    * remove_cluster : bundle에서 cluster 단위로 waypoints를 제거
    * append_cluster : bundle에서 cluster 단위로 waypoints를 추가
    * sort_waypoints_with_two_opt : two-opt 알고리즘을 이용하여 waypoints의 방문 순서를 이동거리가 최소화되도록 정렬
    * update_waypoints_distance_and_time : waypoints를 순서대로 방문하는 조건으로 waypoints간 거리와 이동시간 계산
* BundlePostProcessor : 묶음 재조정 로직을 관리하는 객체
    * relocate_outliers : bundle간 waypoints를 cluster단위로 이동하여 묶음 내 waypoints의 지리적 밀집도를 개선
    * process_bundles : 10초 제한시간동안 relocate_outliers를 최대 5회 반복

### 실행 환경 및 방법

* python3.x
* requirements
```
pip install -r requirements.txt
```

* 실행
```
python bundle_processor.py
```

### sample data
* 서초구의 단일 허브에서 출발하여 서울 및 인근 지역으로 당일배송하는 주문을 29명의 배송원에게 배송시간 기준으로 균등분배하도록 초벌계산한 31개의 배송 묶음

### Tips

* 시간 데이터는 초 단위의 unix timestamp를 사용하며, 알고리즘에서 계산되는 시간은 현재 시각을 0초로 하는 상대시간
* 주요 물리량 단위
	* 시간 - 초
	* 거리 - km
	* 부피 - L
	* 무게 - kg
	* 속도 - km/h
