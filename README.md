# Taiwan Air Quality Index Analysis (2016–2024)

## Mục lục
- [1. Tổng quan dự án và thông tin nhóm](#1-tổng-quan-dự-án-và-thông-tin-nhóm)
- [2. Nguồn và mô tả dữ liệu](#2-nguồn-và-mô-tả-dữ-liệu)
  - [2.1 Chủ đề dữ liệu](#21-chủ-đề-dữ-liệu)
  - [2.2 Nguồn dữ liệu](#22-nguồn-dữ-liệu)
  - [2.3 Giấy phép sử dụng](#23-giấy-phép-sử-dụng)
  - [2.4 Phương pháp thu thập dữ liệu](#24-phương-pháp-thu-thập-dữ-liệu)
  - [2.5 Lý do lựa chọn dữ liệu](#25-lý-do-lựa-chọn-dữ-liệu)
- [3. Câu hỏi nghiên cứu](#3-câu-hỏi-nghiên-cứu)
- [4. Kết quả chính và ý nghĩa thực tiễn](#4-kết-quả-chính-và-ý-nghĩa-thực-tiễn)
- [5. Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
- [6. Hướng dẫn chạy dự án](#6-hướng-dẫn-chạy-dự-án)
- [7. Thư viện và công nghệ sử dụng](#7-thư-viện-và-công-nghệ-sử-dụng)
- [8. Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Tổng quan dự án và thông tin nhóm

### Tên dự án
Phân tích Chỉ số Chất lượng Không khí (AQI) và các chất ô nhiễm chính tại Đài Loan (2016–2024)

### Mục tiêu
Dự án nhằm phân tích dữ liệu Chỉ số Chất lượng Không khí (AQI) tại Đài Loan trong giai đoạn 2016–2024 với các mục tiêu:

- Phân tích biến động ô nhiễm theo thời gian và không gian
- Xác định các chất ô nhiễm chính ảnh hưởng đến rủi ro sức khỏe
- Đánh giá vai trò của tốc độ và hướng gió đối với AQI
- Đề xuất chiến lược giám sát và cảnh báo sớm dựa trên dữ liệu

### Thành viên nhóm
- Thái Khắc Anh Tuấn – 23120112  
- Nguyễn Minh Quân – 23120160  
- Nguyễn Hoàng Minh Trí – 23120179  

---

## 2. Nguồn và mô tả dữ liệu

### 2.1 Chủ đề dữ liệu
Bộ dữ liệu ghi nhận chỉ số AQI và nồng độ các chất ô nhiễm không khí tại các trạm quan trắc trên toàn Đài Loan trong giai đoạn 2016–2024.

Các nhóm biến chính bao gồm:
- Chỉ số tổng hợp: `aqi`, `status`, `pollutant`
- Chất ô nhiễm: `pm2.5`, `pm10`, `o3`, `co`, `so2`, `no`, `no2`, `nox`
- Biến chuẩn AQI: `pm2.5_avg`, `pm10_avg`, `o3_8hr`, `co_8hr`, `so2_avg`
- Điều kiện gió: `windspeed`, `winddirec`
- Thông tin không gian: `sitename`, `county`, `longitude`, `latitude`, `siteid`

Chỉ số AQI phản ánh mức độ rủi ro sức khỏe do ô nhiễm không khí, hỗ trợ công tác cảnh báo y tế cộng đồng, quản lý môi trường và hoạch định chính sách.


### 2.2 Nguồn dữ liệu
- Nền tảng: Kaggle  
- Dataset: Taiwan Air Quality Index Data 2016–2024  
- URL: https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024  

Dữ liệu được tổng hợp từ Bộ Môi trường Đài Loan (Ministry of Environment), với tần suất theo giờ tại các trạm quan trắc.


### 2.3 Giấy phép sử dụng
- Dataset trên Kaggle được cấp phép CC0 – Public Domain
- Dữ liệu gốc của chính phủ Đài Loan tuân theo Open Government Data License v1.0

Bộ dữ liệu được phép sử dụng cho mục đích học tập và nghiên cứu.


### 2.4 Phương pháp thu thập dữ liệu
- Phương pháp: Trạm quan trắc môi trường
- Phạm vi: Toàn Đài Loan
- Thời gian: 2016–2024
- Tần suất: Theo giờ

Hạn chế và sai lệch tiềm năng:
- Phân bố trạm quan trắc không đồng đều, tập trung nhiều ở khu vực đô thị
- Thiếu dữ liệu cục bộ do bảo trì thiết bị hoặc hỏng cảm biến


### 2.5 Lý do lựa chọn dữ liệu
- Ô nhiễm không khí là vấn đề sức khỏe cộng đồng nghiêm trọng tại châu Á
- Bộ dữ liệu có nhiều biến, phù hợp cho phân tích đa biến và mô hình hóa
- Hỗ trợ đa dạng câu hỏi nghiên cứu theo thời gian, không gian và chính sách


## 3. Câu hỏi nghiên cứu

### Q1. Phân tích theo thời gian
- Biến động của PM2.5, O3 và NOx theo giờ trong ngày, ngày trong tuần và mùa trong năm
- So sánh hành vi giữa chất ô nhiễm giao thông (NOx) và chất ô nhiễm thứ cấp (O3)
- Ảnh hưởng của giờ cao điểm giao thông

### Q2. Các đợt ô nhiễm không khí
- Định nghĩa đợt ô nhiễm: AQI > 100 liên tục từ 48 giờ trở lên
- Trạm nào ghi nhận nhiều đợt ô nhiễm kéo dài nhất
- Mối liên hệ giữa các đợt ô nhiễm và yếu tố mùa vụ

### Q3. Ưu tiên cảm biến đo lường
Trong điều kiện chỉ lắp đặt từ 2 đến 3 cảm biến, nên ưu tiên đo chất nào để phản ánh AQI hiệu quả nhất?

Tiêu chí đánh giá:
- Mức độ tương quan với AQI
- Khả năng phân tách theo trạng thái chất lượng không khí
- Tính ổn định trên nhiều trạm quan trắc

### Q4. Mối quan hệ giữa gió và chất lượng không khí
- Ảnh hưởng của tốc độ gió đến AQI
- Mối liên hệ giữa hướng gió và AQI cao hoặc thấp
- Tính ổn định của các kết luận theo không gian

### Q5. Xu hướng và tính mùa vụ
- Xu hướng dài hạn của AQI và PM2.5 trong giai đoạn 2016–2024
- Chu kỳ lặp lại theo năm và theo tháng

---

## 4. Kết quả chính và ý nghĩa thực tiễn

### PM2.5 là chỉ số đại diện tốt nhất cho AQI
- Hệ số tương quan Pearson với AQI xấp xỉ 0.95
- Đề xuất chiến lược giám sát ưu tiên PM2.5 để tối ưu chi phí

### Vai trò của gió trong cải thiện chất lượng không khí
- Gió chỉ phát huy hiệu quả rõ rệt khi tốc độ lớn hơn 3.8–4.3 m/s
- Gió hướng Tây, Tây Bắc và Bắc thường gắn với AQI cao
- Gió hướng Nam thường gắn với AQI thấp

### Tính mùa vụ của ô nhiễm không khí
- Mùa đông (tháng 10–3) xuất hiện các đợt ô nhiễm kéo dài, trung bình khoảng 75 giờ mỗi đợt
- Mùa hè có AQI thấp và ổn định hơn

### Dự báo AQI ngắn hạn
- Mô hình XGBoost đạt MAE khoảng 3.88 và RMSE khoảng 5.55
- Khả năng dự báo đáng tin cậy trong 6 giờ tiếp theo

---

## 5. Cấu trúc thư mục

```bash

├── notebooks/
│   ├── 01_data_collection_and_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_research_question.ipynb
│   └── 04_project_summary.ipynb
│
├── src/
│   ├── collection_and_exploration.py
│   ├── preprocess.py
│   └── viz.py
|
├── environment.yml
├── .gitignore
├── LICENSE
└── README.md
```
Thư mục dự án được tổ chức theo hướng tách biệt rõ ràng giữa **phân tích tương tác (notebooks)** và **mã nguồn tái sử dụng (src)** nhằm đảm bảo tính rõ ràng, dễ bảo trì và thuận tiện cho việc mở rộng.

### notebooks/
Chứa các Jupyter Notebook phục vụ cho quá trình phân tích dữ liệu, thử nghiệm và trình bày kết quả theo từng giai đoạn của dự án:

- `01_data_collection_and_exploration.ipynb` - Thu thập dữ liệu, kiểm tra cấu trúc ban đầu, thống kê mô tả và khám phá dữ liệu (EDA).

- `02_data_preprocessing.ipynb` - Làm sạch dữ liệu, xử lý giá trị thiếu, chuẩn hóa biến và chuẩn bị dữ liệu cho các bước phân tích và mô hình hóa.

- `03_research_question.ipynb` - Trả lời các câu hỏi nghiên cứu đã đề ra, bao gồm phân tích theo thời gian, không gian, vai trò của gió và lựa chọn cảm biến.

- `04_project_summary.ipynb` - Tổng hợp kết quả, trình bày các phát hiện chính và rút ra ý nghĩa thực tiễn của dự án.

### src/
Chứa mã nguồn Python dùng chung, được tách ra khỏi notebook để tăng khả năng tái sử dụng và giữ cho notebook gọn gàng:

- `collection_and_exploration.py` - Các hàm hỗ trợ thu thập dữ liệu và thực hiện khám phá dữ liệu ban đầu.

- `preprocess.py` - Các hàm tiền xử lý dữ liệu như làm sạch, chuyển đổi và chuẩn hóa.

- `viz.py` - Các hàm trực quan hóa dữ liệu, phục vụ vẽ biểu đồ và trình bày kết quả.

### Các tệp khác
- `environment.yml` - Môi trường Conda với tất cả thư viện cần thiết cho dự án.

- `.gitignore`  - Xác định các tệp và thư mục không được theo dõi bởi Git (ví dụ: file tạm, dữ liệu lớn...).

- `LICENSE` - Thông tin giấy phép của dự án.

- `README.md` - Tài liệu mô tả tổng quan dự án, dữ liệu, phương pháp, kết quả và hướng dẫn sử dụng.

## 6. Hướng dẫn chạy dự án

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd <project-folder>
```
### Bước 2: Tạo môi trường Conda
```bash
conda env create -f environment.yml
conda activate min_ds-env
```

### Bước 3: Chạy notebook theo thứ tự

- 01_data_collection_and_exploration.ipynb

- 02_data_preprocessing.ipynb

- 03_Question.ipynb

- 04_project_summary.ipynb

## 7. Thư viện và công nghệ sử dụng

- **Python**: 3.11.5

- **Core & Data Handling**: 
  - pandas 2.1.1  
  - numpy 1.26.0  
  - scipy 1.11.3  
  - openpyxl 3.1.2  

- **Machine Learning**: 
  - scikit-learn 1.3.1  
  - xgboost 1.7.7  

- **Visualization**: 
  - matplotlib 3.8.0  
  - seaborn 0.13.0  
  - plotly 5.17.0  

- **Jupyter & Extensions**: 
  - JupyterLab 4.0.7  
  - jupyter_contrib_nbextensions 0.7.0  
  - ipywidgets 8.1.1  
  - nbdime 3.2.1  

- **Web & Requests**: 
  - requests 2.31.0  
  - requests-cache 1.1.1  
  - beautifulsoup4 4.12.2  
  - selenium 4.15.2  
  - lxml 4.9.3  

- **Version Control & Utilities**: 
  - git 2.42.0  
  - pip 23.3  
  - jupyter-black (formatter)

## 8. Tài liệu tham khảo

- Kaggle: Taiwan Air Quality Index Data 2016–2024
- Ministry of Environment, Taiwan – Open Government Data