import os
import json
import csv

def find_file_in_dir(target_name, search_root):
    """Tìm đường dẫn đầy đủ của một file dựa trên tên file trong một thư mục gốc"""
    for root, dirs, files in os.walk(search_root):
        if target_name in files:
            return os.path.join(root, target_name)
    return None

def get_json_data(filepath):
    """Đọc dữ liệu từ file JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                "status": data.get("status", "N/A"),
                "objective": data.get("objective", "N/A"),
                "time": data.get("time", "N/A")
            }
    except Exception:
        return None

def process_results():
    # Xác định đường dẫn thư mục summary.py đang đứng
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tardiness_root = os.path.join(current_dir, 'result', 'tardiness')
    cost_root = os.path.join(current_dir, 'result', 'cost')
    
    if not os.path.exists(tardiness_root) or not os.path.exists(cost_root):
        print("Lỗi: Không tìm thấy thư mục result/tardiness hoặc result/cost")
        return

    results = []

    # Quét tất cả file .json trong folder tardiness (bao gồm cả N10, N20...)
    print("Đang quét dữ liệu...")
    for root, dirs, files in os.walk(tardiness_root):
        for filename in files:
            if filename.endswith('.json'):
                path_tardiness = os.path.join(root, filename)
                
                path_cost = find_file_in_dir(filename, cost_root)
                
                if path_cost:
                    data_t = get_json_data(path_tardiness)
                    data_c = get_json_data(path_cost)
                    
                    if data_t and data_c:
                        results.append([
                            filename,
                            data_t['objective'],  
                            data_c['objective'],  
                            data_t['status'],     
                            data_c['status'],     
                            data_t['time'],       
                            data_c['time']        
                        ])
                else:
                    print(f"Cảnh báo: Không tìm thấy file đối ứng cho {filename} trong thư mục cost")

    # Ghi dữ liệu ra CSV
    output_path = os.path.join(current_dir, 'summary.csv')
    header = ['Name', 'tardiness', 'cost', 'status_tardiness', 'status_cost', 'time_tardiness', 'time_cost']
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print("-" * 30)
    print(f"Hoàn thành!")
    print(f"Tổng số file đã khớp: {len(results)}")
    print(f"File kết quả: {output_path}")

if __name__ == "__main__":
    process_results()