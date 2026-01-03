# Bộ Dữ liệu VRPBTW Hỗn hợp (Truck-Drone, Backhauls, Time Windows)

Bộ dữ liệu này được thiết kế để kiểm thử các thuật toán giải quyết Bài toán Định tuyến Phương tiện với Phân phối ngược và Cửa sổ thời gian, sử dụng phương tiện hỗn hợp (Truck và Drone).

Các trường hợp được tạo ra dựa trên các yếu tố đa dạng (Vị trí Depot, Phân bố Khách hàng, Tỷ lệ Linehaul/Backhaul) để mô phỏng các kịch bản thực tế khác nhau.

---

## 1. Quy ước Đặt tên File

Tất cả các file dữ liệu được định dạng theo cấu trúc sau:

`[Seed]_[N]_[Depot]_[Dist]_[Ratio].json`

| Ký hiệu | Mô tả | Giá trị Ví dụ |
| :--- | :--- | :--- |
| **Seed** | Seed ngẫu nhiên để tái tạo dữ liệu. | `S042`, `S100` |
| **N** | Số lượng Khách hàng (không bao gồm Depot). | `N50`, `N150` |
| **Depot** | Vị trí Depot. | `C` (Corner: [0, 0]), `Z` (Center: [50, 50]) |
| **Dist** | Phân bố Tọa độ. | `U` (Uniform), `3G`, `5G`, `8G` |
| **Ratio** | Tỷ lệ phần trăm Khách hàng Linehaul (Phân phối đi). | `R50` (50% Linehaul), `R80` (80% Linehaul) |

**Ví dụ:** `S042_N50_C_3G_R50.json`

---

## 2. Thông số Cấu hình Cố định (Tham số Gốc)

Các tham số này được định nghĩa trong file `config.json` và được đưa vào phần `Config` của file JSON dữ liệu:

| Thuộc tính | Đơn vị | Giá trị |
| :--- | :--- | :--- |
| **Khu vực phục vụ** | km | [0, 100] x [0, 100] |
| **Tải trọng Truck** | Đơn vị | 200 (N≤50) / 1000 (N>50) |
| **Tải trọng Drone** | Đơn vị | 50 |
| **Vận tốc Truck** | km/h | 40 |
| **Vận tốc Drone** | km/h | 60 |
| **Thời gian Setup Drone** | Giờ (h) | 1 phút cất cánh + 1 phút hạ cánh (0.0333 h) |
| **Yêu cầu (Demand)** | Đơn vị | Linehaul: [1, 50], Backhaul: [-50, -1] |
| **Phân bổ phương tiện** | Số lượng | Thay đổi theo N (2 cho N10, 10 cho N50, 20 cho N200) |

---

## 3. Cấu trúc File JSON Dữ liệu

Mỗi file JSON chứa dữ liệu tối giản (chỉ dữ liệu gốc) và được chia thành hai phần chính: `Config` và `Nodes`.

### A. Config (Metadata và Hằng số)

Phần này chứa các hằng số và thông tin chung cần thiết cho Solver.

| Khóa | Mô tả |
| :--- | :--- |
| `Config.General` | Thông tin về kích thước bài toán (N, số cụm, T_MAX_SYSTEM). |
| `Config.Vehicles` | Vận tốc, tải trọng, và số lượng phương tiện. |
| `Config.Depot` | Thông tin Depot (Node 0): Tọa độ và Cửa sổ thời gian [0, T_MAX_SYSTEM]. |

### B. Nodes (Thông tin Khách hàng)

Là một danh sách các đối tượng, mỗi đối tượng đại diện cho một khách hàng. Depot (id 0) được định nghĩa trong `Config.Depot`.

| Thuộc tính | Mô tả |
| :--- | :--- |
| `id` | Chỉ số khách hàng (1-based). |
| `coord` | Tọa độ [x, y] (km). |
| `demand` | Yêu cầu của khách hàng (Đơn vị). Dương cho Linehaul, Âm cho Backhaul. |
| `type` | Phân loại: "LINEHAUL" hoặc "BACKHAUL". |
| `tw_h` | Cửa sổ thời gian [t_start, t_end] tính bằng **Giờ**. |

---

## 4. Đặc tả Phân bố Tọa độ

| Ký hiệu | Loại Phân bố | Mô tả |
| :--- | :--- | :--- |
| **U** | Uniform (Đồng nhất) | Tọa độ được sinh ngẫu nhiên đồng nhất trong khu vực [0, 100] x [0, 100]. |
| **$\{K\}G$** | K-Gaussian (Đa cụm) | Khách hàng được nhóm thành $K$ cụm. Số cụm $K$ được xác định tự động: 3 (N≤50), 5 (N≤100), 8 (N>100).  |

Các bộ dữ liệu với **Depot ở Góc (C)** thường tạo ra các tuyến đường dài hơn, ít tối ưu hóa khoảng cách hơn so với các bộ dữ liệu **Depot ở Trung tâm (Z)**.


## Guide
```bash
python vrpbd.py --filename S042_N5_C_3G_R50.json
```
```
