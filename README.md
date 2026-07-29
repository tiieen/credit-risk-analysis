# Phân tích rủi ro tín dụng & dự đoán nợ xấu (Credit Risk Analysis)

Dự án cá nhân — Data Analyst, lĩnh vực ngân hàng.

## Mục tiêu
Phân tích 1.000 hồ sơ vay tín dụng cá nhân để xác định các nhóm khách hàng có rủi ro vỡ nợ cao, xây dựng mô hình dự đoán, và đề xuất khuyến nghị cho bộ phận thẩm định tín dụng.

## Dữ liệu
[German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI Machine Learning Repository) — 1.000 hồ sơ, 20 thuộc tính.

## Nội dung repo
| File | Mô tả |
|---|---|
| `GermanCredit.csv` | Dữ liệu gốc |
| `analysis.py` | Script làm sạch dữ liệu, EDA, xây mô hình Logistic Regression |
| `make_charts.py` | Script tạo biểu đồ cho báo cáo |
| `Credit_Risk_Analysis.ipynb` | Notebook đầy đủ (code + giải thích + biểu đồ) |
| `results.json` | Kết quả phân tích dạng dữ liệu |
| `Bao_cao_Phan_tich_Rui_ro_Tin_dung.docx` | Báo cáo hoàn chỉnh cho stakeholder (Word) |

## Kết quả chính
- Tỷ lệ vỡ nợ tổng thể: **30%**
- Yếu tố ảnh hưởng mạnh nhất: tình trạng tài khoản thanh toán, lịch sử tiết kiệm
- Nhóm rủi ro cao: khách hàng 18-25 tuổi, vay mục đích đào tạo lại/mua ô tô mới
- Mô hình Logistic Regression: **AUC = 0.751**

## Công cụ sử dụng
Python (pandas, scikit-learn, matplotlib, seaborn), SQL-style groupby analysis, Jupyter Notebook

## Cách chạy lại
```bash
pip install pandas scikit-learn matplotlib seaborn
python analysis.py       # chạy phân tích + xuất results.json
python make_charts.py    # xuất biểu đồ PNG
jupyter notebook Credit_Risk_Analysis.ipynb   # xem notebook đầy đủ
```

## Tác giả
Võ Huỳnh Thủy Tiên- email: vohuynhthuytien1769@gmail.com
Portfolio project chuẩn bị cho vị trí Data Analyst / Business Analyst ngân hàng.
