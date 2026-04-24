# Phân tích Hiệu suất Chuyến bay trên DataBricks

## Tổng quan

Dự án phân tích dữ liệu chuyến bay đúng giờ/trễ giờ (On-Time Flight Performance) trên Databricks bằng Apache Spark, GraphFrames và NetworkX.

Mục tiêu chính:
- Mô hình hóa mạng chuyến bay theo dạng đồ thị với sân bay là đỉnh và chuyến bay là cạnh.
- Phân tích các tuyến bay có độ trễ cao, các điểm trung chuyển, và mức độ quan trọng của sân bay.
- Tìm đường đi ngắn theo số chặng giữa các thành phố bằng BFS.
- Trực quan hóa mạng bay bằng biểu đồ mạng và bản đồ địa lý tương tác.

## Nguồn dữ liệu

- OpenFlights: dữ liệu sân bay, hãng bay và tuyến bay.
- US DOT - Bureau of Transportation Statistics (TranStats): dữ liệu chuyến bay và độ trễ.

Khoảng dữ liệu sử dụng trong notebook: từ 01/01/2014 đến 31/03/2014.

## Công nghệ sử dụng

- Apache Spark (PySpark)
- GraphFrames
- NetworkX
- Plotly, Matplotlib
- Pandas, Requests

## Nội dung chính trong notebook

1. Chuẩn bị dữ liệu
- Nạp dữ liệu từ DBFS:
  - `/databricks-datasets/flights/departuredelays.csv`
  - `/databricks-datasets/flights/airport-codes-na.txt`
- Tạo các bảng tạm `airports_na`, `departureDelays`, `tripIATA`, `airports`.

2. Tạo bảng chuẩn hóa chuyến bay
- Xây dựng `departureDelays_geo` gồm thời gian bay, phút trễ, khoảng cách, nguồn/đích và thông tin thành phố, bang.

3. Xây dựng đồ thị chuyến bay
- Vertices: sân bay với khóa `id` (IATA).
- Edges: chuyến bay với `src`, `dst`, `delay`, `tripid`.
- Sử dụng GraphFrames và bổ sung wrapper NetworkX để thực hiện các thuật toán đồ thị ổn định trong môi trường Databricks serverless.

4. Truy vấn phân tích cơ bản
- Đếm số sân bay và số chuyến bay.
- Tìm mức trễ lớn nhất.
- So sánh số chuyến đúng giờ/đến sớm với số chuyến trễ.
- Phân tích các tuyến bay dễ trễ (ví dụ các tuyến xuất phát từ SFO hoặc SEA).

5. Phân tích cấu trúc mạng
- Tính degree, in-degree, out-degree của sân bay.
- Motif analysis cho mẫu `(a)-[ab]->(b); (b)-[bc]->(c)` với trọng tâm trung chuyển qua SFO.
- Xếp hạng sân bay theo PageRank.

6. Phân tích tuyến bay phổ biến và trung chuyển
- Top tuyến bay thẳng phổ biến nhất theo số lượt bay.
- Tính `degreeRatio = inDegree / outDegree` để nhận diện sân bay có tính chất trung chuyển.

7. BFS tìm đường đi theo số chặng
- Kiểm tra tuyến trực tiếp SEA -> SFO.
- Kiểm tra SFO -> BUF (không có chuyến thẳng).
- Tìm tuyến SFO -> BUF có 1 điểm trung chuyển và xếp hạng theo tổng độ trễ thấp nhất.

8. Trực quan hóa
- Trực quan hóa mạng bay bằng NetworkX + Matplotlib, tô màu theo vùng địa lý.
- Các chế độ hiển thị:
  - Chuyến đúng giờ/đến sớm.
  - Chỉ các hub lớn (lọc theo số kết nối).
  - Chuyến bay trễ từ bờ Tây (CA, OR, WA).
  - Toàn bộ mạng bay trong bộ dữ liệu.
- Trực quan hóa bản đồ địa lý tương tác bằng Plotly (dựa trên tọa độ sân bay từ OpenFlights).

## Cài đặt phụ thuộc trong Databricks

Trong notebook có các cell cài đặt trực tiếp:
- `%pip install graphframes`
- `%pip install networkx`
- `%pip install plotly kaleido`

Ngoài ra notebook dùng thêm:
- `matplotlib`
- `pandas`
- `requests`

## Hướng dẫn chạy

1. Tạo cluster Databricks có Spark runtime phù hợp.
2. Import notebook vào workspace Databricks.
3. Chạy notebook từ trên xuống dưới theo thứ tự cell.
4. Khi dùng các cell `%pip install`, khởi động lại Python kernel nếu Databricks yêu cầu.

## Cấu trúc repository

- `On-Time Flight Performance (Spark 2.0).ipynb`: Notebook phân tích chuyến bay.
- `README.md`: Tài liệu mô tả dự án.

## Gợi ý mở rộng

- Thêm dashboard tổng hợp KPI độ trễ theo thời gian thực.
- Phân tích theo mùa, ngày trong tuần và khung giờ cao điểm.
- Kết hợp thêm mô hình dự báo trễ dựa trên thời tiết và lịch sử tuyến bay.