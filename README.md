# AG News Classification - Machine Learning Project (CO3117)

Dự án bài tập lớn môn **Học máy (Machine Learning - CO3117)**: Xây dựng và thực nghiệm hệ thống phân loại văn bản tin tức đa lớp (Multi-class Text Classification) trên bộ dữ liệu **AG News** bằng các thuật toán **Machine Learning truyền thống** (Traditional Machine Learning), kết hợp kỹ thuật trích xuất và lựa chọn đặc trưng (Feature Representation & Feature Selection).

---

## 📑 Mục Lục
1. [Tổng Quan Bài Toán & Dữ Liệu](#1-tổng-quan-bài-toán--dữ-liệu)
2. [Cấu Trúc Thư Mục Dự Án](#2-cấu-trúc-thư-mục-dự-án)
3. [Môi Trường & Hướng Dẫn Cài Đặt](#3-môi-trường--hướng-dẫn-cài-đặt)
   - [Cách 1: Sử dụng Docker (Khuyến nghị để tái lập)](#cách-1-sử-dụng-docker-khuyến-nghị-cho-tính-tái-lập)
   - [Cách 2: Cài đặt trực tiếp môi trường Local (pip / venv)](#cách-2-cài-đặt-trực-tiếp-môi-trường-local-pip--venv)
4. [Tải Dữ Liệu (Dataset Setup)](#4-tải-dữ-liệu-dataset-setup)
5. [Quy Trình Thực Nghiệm & Cách Chạy](#5-quy-trình-thực-nghiệm--cách-chạy)
6. [Các Thuật Toán & Kỹ Thuật Triển Khai](#6-các-thuật-toán--kỹ-thuật-triển-khai)
7. [Thước Đo Đánh Giá & Lý Do Lựa Chọn](#7-thước-đo-đánh-giá--lý-do-lựa-chọn)
8. [Kết Quả Thực Nghiệm & Phân Tích](#8-kết-quả-thực-nghiệm--phân-tích)
9. [Tính Tái Lập & Lưu Trữ Mô Hình](#9-tính-tái-lập--lưu-trữ-mô-hình)
10. [Quy Định & Ghi Chú Nộp Bài](#10-quy-định--ghi-chú-nộp-bài)

---

## 1. Tổng Quan Bài Toán & Dữ Liệu

- **Bài toán**: Phân loại văn bản tin tức tiếng Anh vào 4 chủ đề tin tức độc lập.
- **Tập dữ liệu**: **AG News Dataset** (nguồn chuẩn từ [Kaggle](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)).
- **4 Nhãn Phân Loại (Phân phối cân bằng hoàn hảo - 25% mỗi lớp)**:
  1. `World` (Tin tức thế giới)
  2. `Sports` (Thể thao)
  3. `Business` (Kinh doanh / Tài chính)
  4. `Sci/Tech` (Khoa học & Công nghệ)
- **Kích thước dữ liệu**:
  - Tập huấn luyện thô (`train.csv`): **120.000 mẫu** (30.000 mẫu/nhãn).
  - Tập kiểm thử thô (`test.csv`): **7.600 mẫu** (1.900 mẫu/nhãn).
  - Tập Validation: Tách **18.000 mẫu** từ tập huấn luyện (giữ nguyên tỷ lệ cân bằng) để kiểm chứng mô hình và tinh chỉnh siêu tham số, chống rò rỉ dữ liệu (No Data Leakage).
- **Định hướng thuật toán**: Dự án tập trung hoàn toàn vào **Machine Learning cổ điển** (Linear Models, SVM, Naive Bayes, Feature Selection), tối ưu hóa tốc độ huấn luyện/dự đoán, chi phí tài nguyên thấp và khả năng giải thích cao (interpretable), không sử dụng Deep Learning/Neural Networks phức tạp.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
btl/
├── Dockerfile                         # Khởi tạo container Docker chuẩn hóa môi trường
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── README.md                          # Hướng dẫn chi tiết chạy dự án và kết quả
├── ML_Report_252.pdf                  # Báo cáo tổng kết dự án
├── data/
│   ├── raw/                           # Thư mục chứa dữ liệu gốc (train.csv, test.csv)
│   └── processed/                     # Dữ liệu đã làm sạch và ma trận đặc trưng
├── notebooks/                         # 4 Notebooks theo luồng pipeline tuần tự
│   ├── 01_data_analysis.ipynb         # Khám phá và phân tích đặc trưng dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb         # Tiền xử lý văn bản, trích xuất BoW & TF-IDF
│   ├── 03_model_training.ipynb        # Huấn luyện và đánh giá trên tập Validation
│   └── 04_final_test_evaluation.ipynb # Đánh giá mô hình tốt nhất trên tập Test độc lập
├── src/                               # Mã nguồn module hóa (Modular Architecture)
│   ├── __init__.py
│   ├── data_loader.py                 # Nạp, kiểm tra và tiền xử lý sơ bộ dữ liệu
│   ├── preprocessing.py               # Làm sạch ký tự, tokenize, lemmatize, vectorizer
│   ├── evaluation.py                  # Tính toán các chỉ số và vẽ biểu đồ Confusion Matrix
│   ├── utils.py                       # Lưu trữ / nạp model (.pkl) và các tiện ích
│   └── models/
│       ├── __init__.py
│       ├── baseline.py                # Dummy Classifier, Naive Bayes, Logistic Regression
│       └── traditional_ml.py          # Support Vector Machine (LinearSVC, SVC), Decision Tree, Random Forest
├── experiments/                       # Thí nghiệm mở rộng
│   ├── feature_selection.py           # Lựa chọn đặc trưng (Chi-Square, Mutual Information, Tree-based)
│   └── hyperparameter_tuning.py       # Tinh chỉnh siêu tham số (GridSearchCV, RandomizedSearchCV)
├── models/
│   ├── model_links.txt                # Link Google Drive / Cloud lưu trữ trọng số mô hình lớn
│   └── *.pkl                          # Các mô hình đã huấn luyện (được lưu sau khi chạy notebook)
└── results/
    ├── comparison.csv                 # Bảng tổng hợp so sánh kết quả các mô hình
    └── figures/                       # Các biểu đồ trực quan hóa (Confusion Matrix, phân phối nhãn)
```

---

## 3. Môi Trường & Hướng Dẫn Cài Đặt

Dự án cung cấp cả 2 hình thức triển khai nhằm đảm bảo tính **tái lập kết quả (Reproducibility)** theo đúng yêu cầu đề bài.

### Cách 1: Sử dụng Docker (Khuyến nghị cho tính tái lập)

Docker đảm bảo mã nguồn chạy đồng nhất trên mọi hệ điều hành (Windows, macOS, Linux) mà không gặp xung đột thư viện.

1. **Cài đặt Docker**: Đảm bảo máy tính đã cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. **Build Docker Image**:
   Mở terminal tại thư mục gốc của dự án (`btl/`) và chạy:
   ```bash
   docker build -t ag-news-ml-project .
   ```
3. **Chạy Docker Container với Jupyter Notebook**:
   - **Trên Windows (PowerShell)**:
     ```powershell
     docker run -p 8888:8888 -v ${PWD}:/app ag-news-ml-project
     ```
   - **Trên Linux / macOS**:
     ```bash
     docker run -p 8888:8888 -v $(pwd):/app ag-news-ml-project
     ```
   - **Trên Windows (Command Prompt - cmd)**:
     ```cmd
     docker run -p 8888:8888 -v %cd%:/app ag-news-ml-project
     ```
4. **Truy cập Jupyter Notebook**:
   Mở trình duyệt web và truy cập: `http://localhost:8888` (sao chép mã token hiển thị trên terminal nếu được yêu cầu).

---

### Cách 2: Cài đặt trực tiếp môi trường Local (pip / venv)

**Yêu cầu hệ thống:**
- Python 3.8 đến Python 3.12.
- Quản lý gói `pip`.

**Các bước cài đặt:**

1. **Tạo và kích hoạt môi trường ảo (Virtual Environment)**:
   - Trên Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên Linux / macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2. **Cài đặt các gói phụ thuộc**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Tải tài nguyên NLTK (Punkt, Stopwords, WordNet)**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Khởi động Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

---

## 4. Tải Dữ Liệu (Dataset Setup)

Tập dữ liệu AG News có thể được tải từ:
- [Kaggle AG News Classification Dataset](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)

Sau khi tải, giải nén và đặt 2 file vào đúng vị trí thư mục:
```text
btl/
└── data/
    └── raw/
        ├── train.csv   (120.000 dòng)
        └── test.csv    (7.600 dòng)
```

---

## 5. Quy Trình Thực Nghiệm & Cách Chạy

Dự án được thiết kế thành các bước độc lập, rõ ràng theo đúng chuẩn quy trình Machine Learning:

### Bước 1: Khám Phá & Phân Tích Dữ Liệu (EDA)
Mở và thực thi: [`notebooks/01_data_analysis.ipynb`](file:///notebooks/01_data_analysis.ipynb)
- Kiểm tra dữ liệu khuyết thiếu (`missing values`), kích thước tập dữ liệu.
- Phân tích phân phối của 4 lớp (đảm bảo tính cân bằng 25% mỗi lớp).
- Thống kê độ dài văn bản (số ký tự, số từ) trên từng thể loại tin tức.

### Bước 2: Tiền Xử Lý Văn Bản & Trích Xuất Đặc Trưng
Mở và thực thi: [`notebooks/02_preprocessing.ipynb`](file:///notebooks/02_preprocessing.ipynb)
- Ghép văn bản `Title` + `Description` tạo thành văn bản hoàn chỉnh.
- Làm sạch: Chuyển về chữ thường, loại bỏ thẻ HTML, ký tự đặc biệt, URL và số.
- Tokenization, loại bỏ English Stopwords và Lemmatization bằng WordNet.
- Chia dữ liệu huấn luyện và Validation (`train_test_split` với tỷ lệ 85% train - 15% validation, `stratify=y`, `random_state=42`).
- Biểu diễn đặc trưng:
  - **Bag of Words (BoW)**: `CountVectorizer(max_features=5000, ngram_range=(1, 1))`
  - **TF-IDF**: `TfidfVectorizer(max_features=5000, ngram_range=(1, 1))`
- Lưu các ma trận đặc trưng vào `data/processed/`.

### Bước 3: Huấn Luyện & So Sánh Mô Hình
Mở và thực thi: [`notebooks/03_model_training.ipynb`](file:///notebooks/03_model_training.ipynb)
- Thử nghiệm 6 tổ hợp giữa 3 thuật toán và 2 phương pháp biểu diễn đặc trưng:
  1. Naive Bayes + BoW
  2. Naive Bayes + TF-IDF
  3. Logistic Regression + BoW
  4. Logistic Regression + TF-IDF
  5. Linear SVM + BoW
  6. Linear SVM + TF-IDF
- Đánh giá trên tập Validation (18.000 mẫu) và xuất bảng so sánh vào `results/comparison.csv`.
- Lưu các mô hình đã huấn luyện vào `models/`.

### Bước 4: Đánh Giá Kiểm Thử Độc Lập Cuối Cùng
Mở và thực thi: [`notebooks/04_final_test_evaluation.ipynb`](file:///notebooks/04_final_test_evaluation.ipynb)
- Nạp mô hình có hiệu năng cao nhất từ Bước 3 (**Linear SVM + TF-IDF**).
- Đánh giá trên tập kiểm thử độc lập (`data/raw/test.csv` - 7.600 mẫu).
- Trực quan hóa ma trận nhầm lẫn (Confusion Matrix) và phân tích chi tiết độ chính xác từng lớp.

### Bước 5: Các Thí Nghiệm Mở Rộng (Advanced Experiments)
Chạy trực tiếp từ dòng lệnh:
1. **Lựa chọn đặc trưng (Feature Selection)**:
   ```bash
   python experiments/feature_selection.py
   ```
   Thử nghiệm các phương pháp lựa chọn thuộc tính: Chi-Square ($\chi^2$), Mutual Information và Tree-based Selection (Random Forest).
2. **Tinh chỉnh siêu tham số (Hyperparameter Tuning)**:
   ```bash
   python experiments/hyperparameter_tuning.py
   ```
   Sử dụng `GridSearchCV` và `RandomizedSearchCV` (5-Fold Cross Validation) để tìm giá trị tối ưu của tham số điều chuẩn `C` cho SVM và Logistic Regression, tham số `alpha` cho Naive Bayes.

---

## 6. Các Thuật Toán & Kỹ Thuật Triển Khai

| Thuật toán / Kỹ thuật | Phân loại | Mô tả & Vai trò trong dự án |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Baseline Model | Mô hình xác suất dựa trên định lý Bayes với giả định độc lập có điều kiện. Tốc độ cực nhanh (huấn luyện < 0.02s), đóng vai trò mốc so chuẩn ban đầu. |
| **Logistic Regression** | Linear Model | Phân loại tuyến tính với hàm mất mát Cross-Entropy và điều chuẩn $L_2$ ($C=1.0$). Hiệu năng xuất sắc, tốc độ hội tụ nhanh. |
| **Linear SVM (LinearSVC)** | Margin-based | Tìm siêu phẳng tối ưu phân tách các lớp với biên độ lớn nhất. Rất thích hợp với không gian đặc trưng văn bản nhiều chiều và thưa thớt (sparse). |
| **Bag of Words (BoW)** | Feature Representation | Đếm tần suất xuất hiện của từ vựng (`CountVectorizer`), giới hạn 5.000 từ quan trọng nhất. |
| **TF-IDF** | Feature Representation | Đánh trọng số từ dựa trên tần suất từ (TF) và nghịch đảo tần suất văn bản (IDF), giảm trọng số các từ xuất hiện quá phổ biến. |
| **Feature Selection** | Advanced Technique | Sử dụng kiểm định Chi-Square ($\chi^2$) và Mutual Information để trích chọn tập từ vựng mang nhiều thông tin phân loại nhất. |

---

## 7. Thước Đo Đánh Giá & Lý Do Lựa Chọn

Theo đúng yêu cầu của môn học, việc lựa chọn các chỉ số đo lường cần có căn cứ khoa học:

1. **Accuracy (Độ chính xác tổng thể)**:
   - *Lý do*: Bộ dữ liệu AG News có 4 lớp phân bố **cân bằng hoàn hảo** (mỗi lớp chiếm chính xác 25% trong cả tập train và test). Do đó, Accuracy phản ánh khách quan và chính xác hiệu năng toàn diện của mô hình mà không gặp hiện tượng mất cân bằng nhãn (class imbalance).
2. **Macro Precision, Macro Recall & Macro F1-Score**:
   - *Lý do*: Tính trung bình không trọng số trên cả 4 lớp. Điều này đảm bảo đánh giá công bằng năng lực nhận diện của mô hình trên từng thể loại (World, Sports, Business, Sci/Tech), tránh trường hợp mô hình chỉ tối ưu một lớp cụ thể.
3. **Confusion Matrix (Ma trận nhầm lẫn)**:
   - *Lý do*: Trực quan hóa chi tiết các lỗi phân loại sai, giúp xác định các cặp nhãn có sự giao thoa ngữ nghĩa lớn (ví dụ: các bài báo tin tức kinh doanh công nghệ bị nhầm giữa `Business` và `Sci/Tech`).
4. **Thời Gian Huấn Luyện & Dự Đoán (Train / Predict Time)**:
   - *Lý do*: Đánh giá sự đánh đổi (trade-off) giữa độ chính xác và chi phí tài nguyên tính toán trong triển khai thực tế.

---

## 8. Kết Quả Thực Nghiệm & Phân Tích

### 8.1 So Sánh Các Mô Hình Trên Tập Validation (18.000 mẫu)

| Thuật toán | Đặc trưng | Validation Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **TF-IDF** | **91.38%** | **0.9137** | **0.9138** | **0.9136** | 2.79s |
| **Logistic Regression** | **TF-IDF** | 91.27% | 0.9126 | 0.9127 | 0.9125 | 1.89s |
| **Linear SVM** | **BoW** | 90.40% | 0.9039 | 0.9040 | 0.9038 | 7.17s |
| **Logistic Regression** | **BoW** | 90.28% | 0.9026 | 0.9028 | 0.9026 | 1.90s |
| **Naive Bayes** | **TF-IDF** | 89.90% | 0.8987 | 0.8990 | 0.8987 | **0.01s** |
| **Naive Bayes** | **BoW** | 89.87% | 0.8983 | 0.8987 | 0.8984 | **0.02s** |

### 8.2 Kết Quả Kiểm Thử Cuối Cùng Trên Tập Test (7.600 mẫu)

Mô hình tốt nhất (**Linear SVM + TF-IDF**) được kiểm định trên tập dữ liệu Test hoàn toàn độc lập:
- **Test Accuracy**: **90.51%** (suy giảm không đáng kể so với tập validation ~0.86%, chứng tỏ mô hình không bị quá khớp - Overfitting).
- **Macro F1-Score**: **90.49%**
- **Hiệu năng chi tiết theo từng lớp (Per-class F1-Score)**:
  - `Sports`: **96.00%** (Phân loại chính xác nhất nhờ tập từ vựng thể thao mang tính đặc trưng cao: *match, win, game, champion, coach*).
  - `World`: **91.00%** (Độ chính xác cao, nhận diện tốt các thực thể địa chính trị).
  - `Sci/Tech`: **88.00%**
  - `Business`: **87.00%** (Có sự nhầm lẫn nhẹ với `Sci/Tech` do nhiều bài báo về các tập đoàn công nghệ lớn như Apple, Google, Microsoft vừa mang tính kinh doanh vừa mang tính công nghệ).

### 8.3 Nhận Xét & Kết Luận
1. **Biểu diễn TF-IDF vượt trội hơn BoW**: Trọng số TF-IDF giúp giảm thiểu tác động của các từ thông dụng, cải thiện độ chính xác từ 0.8% - 1.0% trên toàn bộ các mô hình.
2. **Linear SVM đạt kết quả tốt nhất**: Phù hợp tối ưu với bài toán phân loại văn bản nhiều chiều, không gian vector thưa.
3. **Naive Bayes là baseline tối ưu**: Đạt độ chính xác xấp xỉ 90% chỉ với 0.01s huấn luyện, cực kỳ hiệu quả khi cần xử lý nhanh với tài nguyên hạn chế.

---

## 9. Tính Tái Lập & Lưu Trữ Mô Hình

- **Ngẫu nhiên hóa có kiểm soát**: Mọi thao tác chia tập dữ liệu (`train_test_split`) và khởi tạo thuật toán đều được cố định `random_state = 42`.
- **Lưu trữ mô hình**:
  - Các file trọng số mô hình dạng `.pkl` được lưu trong thư mục `models/`.
  - Để tránh vượt dung lượng nộp bài, các file mô hình lớn và ma trận vectorizer được lưu trữ tại liên kết công khai trong file [`models/model_links.txt`](file:///models/model_links.txt).

---

## 10. Quy Định & Ghi Chú Nộp Bài

- Dự án tuân thủ đầy đủ quy định nộp bài môn **Học máy (CO3117)**:
  - Đầy đủ mã nguồn sạch, tài liệu hướng dẫn `README.md`, Dockerfile, file môi trường `requirements.txt`.
  - Báo cáo chi tiết định dạng PDF: [`ML_Report_252.pdf`](file:///ML_Report_252.pdf).
  - Không sử dụng các API thương mại đóng kín thay thế pipeline, toàn bộ thuật toán được xây dựng minh bạch bằng các thư viện mã nguồn mở chuẩn mực (`scikit-learn`, `numpy`, `pandas`, `nltk`).
