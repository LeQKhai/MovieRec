Movie Recommender System

Giới thiệu
Movie Recommender System là ứng dụng đề xuất phim dựa trên sở thích người dùng, sử dụng dữ liệu từ MovieLens và TMDb API. Hệ thống triển khai ba phương pháp gợi ý:

Ứng dụng sử dụng Streamlit để tạo giao diện thân thiện, hiển thị gợi ý kèm poster phim.

Cài đặt

Clone repository:
  git clone https://github.com/LeQKhai/MovieRec

Cài đặt thư viện:
  pip install -r requirements.txt
  
Sử dụng
Chạy ứng dụng:
  streamlit run app.py
  
Truy cập giao diện tại http://localhost:8501.

Chọn chế độ:

Gợi ý phim: Chọn một phim và phương pháp (Collaborative, Content-Based, Hybrid) để nhận gợi ý.

Chọn theo thể loại: Chọn thể loại để xem top 10 phim phổ biến.


Cấu trúc mã

app.py: File chính, chứa logic ứng dụng và giao diện Streamlit.

FixTitle.py, FixEncoding.py: Module chuẩn hóa tiêu đề phim.

data2/: Thư mục chứa dữ liệu MovieLens (tự động tải).

Dữ liệu

MovieLens: Bao gồm ratings.csv, movies.csv, tags.csv, links.csv.

TMDb API: Cung cấp poster phim dựa trên tmdbId.
