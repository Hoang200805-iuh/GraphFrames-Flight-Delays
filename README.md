# On-Time Flight Performance với GraphFrames

Đây là một dự án notebook Spark dùng để phân tích độ trễ chuyến bay và cấu trúc mạng lưới hàng không bằng các kỹ thuật đồ thị và học máy.

Repository này được phát triển từ notebook Databricks kinh điển **On-Time Flight Performance with GraphFrames for Apache Spark**, sau đó được chỉnh sửa để có thể chạy cục bộ trong môi trường VS Code Notebook trên Windows với dữ liệu lưu trên máy tính.

## Mục Tiêu Dự Án

Notebook tập trung vào hai hướng chính:

1. **Khai phá đồ thị trên tuyến bay**
   - Mô hình hóa sân bay thành các đỉnh của đồ thị
   - Mô hình hóa chuyến bay thành các cạnh của đồ thị
   - Phân tích độ trễ, mức độ kết nối của sân bay, các hub trung chuyển và các tuyến bay phổ biến

2. **Dự đoán độ trễ bằng Spark ML**
   - Chuẩn bị dữ liệu bay cho bài toán phân loại
   - Xây dựng pipeline đặc trưng
   - Huấn luyện mô hình logistic regression
   - Đánh giá mức độ mô hình dự đoán chuyến bay có bị trễ hay không

## Bộ Dữ Liệu

Dự án sử dụng hai file dữ liệu cục bộ trong thư mục `flight-data/`:

- `airport-codes-na.txt` - thông tin sân bay như thành phố, bang, quốc gia và mã IATA
- `departuredelays.csv` - dữ liệu độ trễ chuyến bay với ngày bay, độ trễ, khoảng cách, điểm đi và điểm đến

### Nguồn dữ liệu

Notebook gốc được lấy từ Databricks và sử dụng dữ liệu chuyến bay nội địa Hoa Kỳ thuộc bộ dữ liệu On-Time Performance.

## Notebook Thực Hiện Gì

### 1. Chuẩn bị dữ liệu
- Nạp dữ liệu sân bay và dữ liệu độ trễ vào Spark DataFrame
- Tạo các SQL view tạm để dùng lại trong các truy vấn phía sau
- Lọc danh sách sân bay chỉ còn các sân bay xuất hiện trong dữ liệu chuyến bay
- Ghép dữ liệu thành bảng tổng hợp gồm tuyến bay, độ trễ, khoảng cách và thông tin vị trí sân bay

### 2. Xây dựng đồ thị
- Chuyển sân bay thành các đỉnh của đồ thị
- Chuyển chuyến bay thành các cạnh của đồ thị
- Tạo đối tượng đồ thị để phân tích
- Tạo một đồ thị nhỏ hơn cho các truy vấn motif

### 3. Phân tích đồ thị
Notebook tính toán và khám phá các thông tin sau:
- Tổng số sân bay và số chuyến bay
- Độ trễ lớn nhất xuất hiện trong dữ liệu
- Số chuyến bị trễ so với số chuyến đúng giờ hoặc đến sớm
- Mẫu độ trễ theo tuyến bay xuất phát từ Seattle
- Degree, in-degree và out-degree của từng sân bay
- Các sân bay có tiềm năng trở thành hub trung chuyển dựa trên tỷ lệ in/out degree
- Đường đi ngắn bằng breadth-first search giữa các sân bay
- Các motif biểu diễn những chuyến bay nối chuyến
- Điểm PageRank để đo mức độ quan trọng của sân bay
- Các tuyến bay trực tiếp phổ biến nhất

### 4. Học máy
Notebook cũng xây dựng một quy trình dự đoán:
- Tạo nhãn cho chuyến bay bị trễ hoặc không bị trễ
- Mã hóa biến phân loại bằng one-hot encoding
- Ghép các đặc trưng số và đặc trưng phân loại thành một vector
- Chia dữ liệu thành tập train và test
- Huấn luyện bộ phân loại logistic regression
- Đánh giá mô hình trên tập test

## Tùy Chỉnh Để Chạy Cục Bộ

Notebook đã được điều chỉnh để chạy local trong VS Code thay vì chỉ chạy trong Databricks.

Các thay đổi chính:

- Dùng `SparkSession` local
- Đọc dữ liệu từ thư mục `flight-data/` trên máy
- Thay đường dẫn DBFS của Databricks bằng đường dẫn Windows cục bộ
- Thêm một lớp tương thích GraphFrame cục bộ nhẹ
- Chuyển các lệnh `print` kiểu Python 2 sang cú pháp Python 3
- Thay một số phần trực quan hóa chỉ có trong Databricks bằng phần giải thích trong notebook

## Yêu Cầu Hệ Thống

Để chạy notebook cục bộ, bạn cần:

- Python 3
- PySpark
- NetworkX
- Hỗ trợ Jupyter Notebook trong VS Code

Notebook đã được phát triển và kiểm thử trong môi trường Windows cục bộ.

## Cách Chạy

1. Mở thư mục repository trong VS Code.
2. Mở file `On-Time Flight Performance (Spark 2.0).ipynb`.
3. Chạy lần lượt các cell từ trên xuống dưới.
4. Đảm bảo thư mục `flight-data/` nằm cùng cấp với notebook.

## Kết Quả Mong Đợi

Notebook sẽ tạo ra các kết quả như:

- Số lượng sân bay và số lượng chuyến bay
- Độ trễ lớn nhất tính theo phút
- Tổng số chuyến bị trễ và không bị trễ
- Xếp hạng các sân bay theo degree
- Các ứng viên hub trung chuyển
- Kết quả BFS giữa các sân bay được chọn
- Xếp hạng PageRank của sân bay
- Kết quả dự đoán bằng logistic regression
- Điểm đánh giá mô hình

## Cấu Trúc Dự Án

```text
On-Time Flight Performance (Spark 2.0).ipynb
flight-data/
  airport-codes-na.txt
  departuredelays.csv
README.md
```

## Ghi Chú

- Notebook gốc có các cell Scala và trực quan hóa D3 đặc thù của Databricks. Trong bản local này, các phần đó được ghi chú rõ ràng nhưng không phụ thuộc vào chúng để chạy.
- Notebook này phục vụ mục đích học tập và phân tích dữ liệu đồ thị, không phải một hệ thống dự đoán độ trễ chuyến bay dùng trong sản xuất.

## Tài Liệu Tham Khảo

- GraphFrames: https://graphframes.github.io/user-guide.html
- Notebook gốc từ Databricks: https://databricks.com/blog/2016/03/16/on-time-flight-performance-with-graphframes-for-apache-spark.html
- OpenFlights: http://openflights.org/data.html
