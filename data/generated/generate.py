import numpy as np
import random
import json
import argparse
import os
import shutil


def load_config(config_path):
    """Tải cấu hình từ file JSON."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def determine_num_clusters(num_customers):
    """Xác định số lượng cụm K dựa trên số lượng khách hàng."""
    if num_customers <= 50:
        return 3
    elif num_customers <= 100:
        return 5
    else:
        return 8


def generate_multi_gaussian_coords(num_customers, max_coord, num_clusters, seed):
    """Sinh tọa độ Multi-Gaussian (đa cụm)."""
    random.seed(seed)
    np.random.seed(seed)

    # 1. Xác định K tâm cụm ngẫu nhiên
    cluster_centers = np.random.uniform(0, max_coord, size=(num_clusters, 2))

    # 2. Xác định độ lệch chuẩn (sigma)
    std_dev = max_coord / (4.0 * num_clusters)

    # 3. Phân bổ khách hàng vào các cụm
    cluster_assignments = np.random.randint(0, num_clusters, size=num_customers)

    customer_coords = np.zeros((num_customers, 2))
    for i in range(num_customers):
        center_idx = cluster_assignments[i]
        center = cluster_centers[center_idx]
        coord = np.random.normal(center, std_dev)
        customer_coords[i] = np.clip(coord, 0, max_coord)

    return customer_coords, num_clusters


def generate_filename(
    num_customers, depot_location, coord_distribution, demand_split_ratio, seed
):
    """Tạo tên file theo quy ước [Seed]_[N]_[Depot]_[Dist]_[Ratio].json"""

    seed_str = f"S{seed:03d}"
    depot_char = "C" if depot_location == "corner" else "Z"

    if coord_distribution == "uniform":
        dist_char = "U"
    else:  # multi-gaussian
        num_clusters = determine_num_clusters(num_customers)
        dist_char = f"{num_clusters}G"

    ratio_int = int(demand_split_ratio * 100)

    filename = f"{seed_str}_N{num_customers}_{depot_char}_{dist_char}_R{ratio_int}.json"
    return filename


def generate_vrpbtw_data(
    config, num_customers, coord_distribution, demand_split_ratio, depot_location, seed
):
    """
    Sinh dữ liệu VRPBTW tối giản (chỉ giữ dữ liệu gốc) theo cấu trúc Config/Nodes.
    (Giữ nguyên logic tính toán)
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- 1. LẤY THÔNG SỐ CẤU HÌNH ---
    MAX_COORD = config["MAX_COORD"]
    V_TRUCK = config["V_TRUCK_KM_H"]

    CAPACITY_TRUCK = (
        config["CAPACITY_MAP"]["SMALL"]
        if num_customers <= config["CAPACITY_MAP"]["THRESHOLD"]
        else config["CAPACITY_MAP"]["LARGE"]
    )
    CAPACITY_DRONE = config["CAPACITY_DRONE"]
    SCALING_FACTOR = config["TIME_WINDOW_SCALING_FACTOR"]

    num_vehicles = config["FLEET_SIZES"].get(str(num_customers), 1)
    num_nodes = num_customers + 1
    num_clusters = 0

    # --- 2. TẠO TỌA ĐỘ ---
    if depot_location == "corner":
        depot_coord = np.array([0.0, 0.0])
    else:
        # depot_location == "center":
        depot_coord = np.array([MAX_COORD / 2.0, MAX_COORD / 2.0])

    if coord_distribution == "uniform":
        customer_coords = np.random.uniform(0, MAX_COORD, size=(num_customers, 2))
    elif coord_distribution == "multi-gaussian":
        num_clusters = determine_num_clusters(num_customers)
        customer_coords, num_clusters = generate_multi_gaussian_coords(
            num_customers, MAX_COORD, num_clusters, seed
        )
    else:
        center = MAX_COORD / 2
        std_dev = MAX_COORD / 4
        customer_coords = np.clip(
            np.random.normal(center, std_dev, size=(num_customers, 2)), 0, MAX_COORD
        )

    coords = np.vstack([depot_coord, customer_coords])

    # --- 3. TẠO YÊU CẦU VÀ TW ---
    L_min, L_max = config["DEMAND_RANGE_LINEHAUL"]
    B_min, B_max = config["DEMAND_RANGE_BACKHAUL"]
    num_linehaul = int(num_customers * demand_split_ratio)
    demands = np.concatenate(
        [
            np.random.randint(L_min, L_max + 1, size=num_linehaul),
            np.random.randint(B_min, B_max + 1, size=num_customers - num_linehaul),
        ]
    )
    np.random.shuffle(demands)
    full_demands = np.insert(demands, 0, 0)

    # Tính T_max
    dist_0_i = np.linalg.norm(coords[1:] - coords[0], ord=1, axis=1)
    D_max = np.sum(dist_0_i) * 2
    T_max = D_max / V_TRUCK

    time_windows = np.zeros((num_nodes, 2))
    time_windows[0] = [0.0, T_max]

    # T_max_per_truck = T_max / num_vehicles
    T_max_per_truck = 24.0  # keep lower bound in a day

    customer_nodes = []
    for i in range(1, num_nodes):
        Delta_i = (np.linalg.norm(coords[i] - coords[0]) * 2) / V_TRUCK
        tstart_i = random.uniform(0, T_max_per_truck)
        random_end_duration = random.uniform(0, SCALING_FACTOR * Delta_i)
        tend_i = tstart_i + random_end_duration
        time_windows[i] = [tstart_i, tend_i]

        node_type = "LINEHAUL" if full_demands[i] > 0 else "BACKHAUL"
        customer_nodes.append(
            {
                "id": i,
                "coord": coords[i].tolist(),
                "demand": int(full_demands[i]),
                "type": node_type,
                "tw_h": time_windows[i].tolist(),
            }
        )

    data = {
        "Config": {
            "General": {
                "NUM_CUSTOMERS": num_customers,
                "NUM_NODES": num_nodes,
                "MAX_COORD_KM": MAX_COORD,
                "T_MAX_SYSTEM_H": T_max,
                "TIME_WINDOW_SCALING_FACTOR": SCALING_FACTOR,
                "COORD_DISTRIBUTION": coord_distribution,
                "NUM_CLUSTERS": num_clusters,
            },
            "Vehicles": {
                "NUM_TRUCKS": int(num_vehicles),
                "NUM_DRONES": int(num_vehicles),
                "V_TRUCK_KM_H": V_TRUCK,
                "V_DRONE_KM_H": config["V_DRONE_KM_H"],
                "CAPACITY_TRUCK": int(CAPACITY_TRUCK),
                "CAPACITY_DRONE": int(CAPACITY_DRONE),
                "DRONE_TAKEOFF_MIN": config["DRONE_TAKEOFF_MIN"],
                "DRONE_LANDING_MIN": config["DRONE_LANDING_MIN"],
                "SERVICE_TIME_MIN": config["SERVICE_TIME_MIN"],
                "DRONE_DURATION_H": config["DRONE_DURATION_H"],
            },
            "Depot": {
                "id": 0,
                "coord": depot_coord.tolist(),
                "time_window_h": time_windows[0].tolist(),
            },
        },
        "Nodes": customer_nodes,
    }
    return data


def run_single_instance(
    config,
    num_customers,
    coord_distribution,
    demand_split_ratio,
    depot_location,
    seed,
    output_dir,
):
    """
    Hàm sinh và lưu một bộ dữ liệu đơn lẻ.
    """

    dataset = generate_vrpbtw_data(
        config=config,
        num_customers=num_customers,
        coord_distribution=coord_distribution,
        demand_split_ratio=demand_split_ratio,
        depot_location=depot_location,
        seed=seed,
    )

    filename = generate_filename(
        num_customers, depot_location, coord_distribution, demand_split_ratio, seed
    )

    instance_dir = os.path.join(output_dir, f"N{num_customers}")
    os.makedirs(instance_dir, exist_ok=True)

    output_path = os.path.join(instance_dir, filename)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"   [OK] Đã lưu: {output_path}")


def run_batch_generation(
    config_path,
    output_dir,
    seed_start,
    num_instances,
    customer_counts,
    ratios,
    depot_locations,
):
    """
    Hàm chạy chính để sinh hàng loạt các bộ dữ liệu.
    """
    config = load_config(config_path)
    print("--- BATCH GENERATION BẮT ĐẦU ---")

    for N in customer_counts:
        print(f"\n--- Sinh bộ dữ liệu N={N} ---")

        for depot in depot_locations:
            for ratio in ratios:
                current_seed = seed_start
                for i in range(num_instances):
                    print(
                        f"  > Cấu hình: Depot={depot}, Ratio={ratio}, Instance={i + 1}/{num_instances}, Seed={current_seed}"
                    )

                    # 1. Sinh trường hợp UNIFORM
                    run_single_instance(
                        config=config,
                        num_customers=N,
                        coord_distribution="uniform",
                        demand_split_ratio=ratio,
                        depot_location=depot,
                        seed=current_seed,
                        output_dir=output_dir,
                    )

                    # 2. Sinh trường hợp MULTI-GAUSSIAN (Clustered)
                    run_single_instance(
                        config=config,
                        num_customers=N,
                        coord_distribution="multi-gaussian",
                        demand_split_ratio=ratio,
                        depot_location=depot,
                        seed=current_seed,
                        output_dir=output_dir,
                    )

                    # Tăng seed lên 1 cho instance tiếp theo
                    current_seed += 1

    print("\n--- BATCH GENERATION HOÀN TẤT ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script sinh hàng loạt dữ liệu VRPBTW tối giản."
    )
    parser.add_argument(
        "--config_path",
        default="config.json",
        type=str,
        help="Đường dẫn đến file cấu hình JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Thư mục gốc lưu trữ dữ liệu. (Mặc định: generated)",
    )

    parser.add_argument(
        "--seed_start",
        type=int,
        default=42,
        help="Seed ngẫu nhiên bắt đầu. Sau mỗi instance sẽ tăng lên 1. Mặc định: 42",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=5,
        help="Số lượng instance cần sinh cho mỗi cấu hình cơ bản. Mặc định: 5",
    )

    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Danh sách số lượng khách hàng (N) cần sinh. Mặc định: 20, 50, 100",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="Danh sách tỷ lệ Linehaul (0.0 đến 1.0). Mặc định: 0.5 (cân bằng), 0.8 (chủ yếu Linehaul)",
    )
    parser.add_argument(
        "--depots",
        type=str,
        nargs="+",
        default=["corner", "center"],
        choices=["corner", "center"],
        help="Danh sách vị trí Depot. Mặc định: 'corner', 'center'",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Lỗi: Không tìm thấy file cấu hình tại {args.config_path}")
        exit()

    if os.path.exists(args.output_dir):
        print(
            f"CẢNH BÁO: Đang tiến hành xóa thư mục '{args.output_dir}' và toàn bộ nội dung..."
        )

        try:
            shutil.rmtree(args.output_dir)
            print(f"Xóa thành công thư mục '{args.output_dir}'.")
        except OSError as e:
            print(f"Lỗi khi xóa thư mục {args.output_dir}: {e}")
    else:
        print(f"Thư mục '{args.output_dir}' không tồn tại. Không cần xóa.")

    run_batch_generation(
        config_path=args.config_path,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        num_instances=args.num_instances,
        customer_counts=args.counts,
        ratios=args.ratios,
        depot_locations=args.depots,
    )
