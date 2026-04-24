# Phân tích Hiệu suất Chuyến bay với GraphFrames trên Databricks

## Tổng quan

Dự án này phân tích dữ liệu chuyến bay đúng giờ/trễ giờ (On-Time Flight Performance) bằng Apache Spark và tư duy đồ thị (GraphFrames).

Notebook chính:
- `On-Time Flight Performance (Spark 2.0).ipynb`

Mục tiêu chính:
- Khám phá mạng lưới sân bay và tuyến bay dưới dạng graph.
- Phân tích độ trễ chuyến bay theo tuyến, theo điểm đi/điểm đến.
- Tìm quan hệ trung chuyển bằng motif và BFS.
- Xếp hạng mức độ quan trọng sân bay bằng PageRank.
- Xây dựng pipeline Machine Learning dự đoán chuyến bay bị trễ.

## Nguồn dữ liệu

- OpenFlights: dữ liệu sân bay/tuyến bay
- US DOT - Bureau of Transportation Statistics (TranStats): dữ liệu chuyến bay và độ trễ

Trong notebook, dữ liệu được sử dụng cho giai đoạn mẫu (khoảng 01/2014 đến 03/2014).

## Công nghệ sử dụng

- Apache Spark (PySpark)
- GraphFrames (hoặc lớp GraphFrame thay thế để chạy local)
- NetworkX (hỗ trợ PageRank trong phiên bản local)
- Spark MLlib Pipeline:
  - StringIndexer
  - OneHotEncoder
  - VectorAssembler
  - LogisticRegression

## Nội dung phân tích trong notebook

1. Chuẩn bị và nạp dữ liệu
- Đọc dữ liệu chuyến bay và dữ liệu sân bay.
- Chuẩn hóa, join dữ liệu để tạo bảng chuyến bay có thông tin địa lý.

2. Xây dựng graph chuyến bay
- Vertices: sân bay (`id`).
- Edges: chuyến bay (`src`, `dst`, `delay`, ...).

3. Truy vấn phân tích cơ bản
- Số lượng sân bay/chuyến bay.
- Độ trễ lớn nhất.
- So sánh chuyến bay đúng giờ và chuyến bay trễ.
- Tuyến bay thường trễ, đặc biệt từ SEA.

4. Phân tích cấu trúc mạng
- Degree, in-degree, out-degree.
- Motif finding cho các hành trình nối chuyến (ví dụ qua SFO).
- PageRank để xếp hạng sân bay quan trọng.

5. Tìm đường đi bằng BFS
- Kiểm tra chuyến trực tiếp và chuyến nối giữa các sân bay.
- Thống kê điểm trung chuyển phổ biến.

6. Dự đoán trễ chuyến bay (ML)
- Tạo đặc trưng, vector hóa dữ liệu.
- Chia train/test.
- Huấn luyện Logistic Regression.
- Đánh giá mô hình bằng BinaryClassificationEvaluator.

## Hướng dẫn chạy trên Databricks

1. Tạo cluster Databricks với runtime có Spark phù hợp.
2. Import notebook `On-Time Flight Performance (Spark 2.0).ipynb` vào workspace.
3. Chuẩn bị dữ liệu đầu vào (CSV/TXT) và cập nhật đường dẫn trong notebook cho phù hợp DBFS.
4. Cài package cần thiết (nếu cần):
- `graphframes`
- `networkx`
5. Chạy notebook theo thứ tự từ trên xuống dưới.

## Chạy local (tùy chọn)

Bạn có thể chạy local với PySpark nếu đã cài Java + Spark + Python và các thư viện tương ứng.

Lưu ý:
- Notebook hiện chứa ví dụ đường dẫn dữ liệu cục bộ; cần đổi lại theo máy của bạn.
- Một số cell đã có lớp GraphFrame giả lập để giảm phụ thuộc môi trường Databricks.

## Cấu trúc repository

- `On-Time Flight Performance (Spark 2.0).ipynb`: Notebook phân tích chính.
- `README.md`: Tài liệu mô tả dự án.

## Định hướng mở rộng

- Bổ sung trực quan hóa mạng bay (Plotly/Kepler/D3).
- Thử thêm mô hình ML khác (Random Forest, GBT, XGBoost).
- Đánh giá theo nhiều chỉ số hơn (AUC, F1, Precision/Recall).
- Đóng gói pipeline thành job tự động trên Databricks.

## Ghi nhận

Dự án được phát triển dựa trên ý tưởng phân tích hiệu suất chuyến bay bằng GraphFrames và được điều chỉnh để chạy thuận tiện hơn trong môi trường local/Databricks.