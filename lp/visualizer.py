import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Load the instance data
data_path = "E:/bkai/VRPB/hrl4vdrpbtw/data/generated/data/N5/S042_N5_C_U_R50.json"
with open(data_path, "r") as f:
    data = json.load(f)

# Extract depot and customer information
depot = data["Config"]["Depot"]
depot_coord = depot["coord"]
customers = data["Nodes"]

# Create mappings
node_coords = {0: depot_coord}
node_info = {0: {"type": "DEPOT", "demand": 0, "coord": depot_coord}}

for customer in customers:
    node_coords[customer["id"]] = customer["coord"]
    node_info[customer["id"]] = {
        "type": customer["type"],
        "demand": customer["demand"],
        "coord": customer["coord"],
    }

# LP solution
vehicle_routes = [
    {
        "route": [0, 3, 5, 1, 4, 0],
        "drone_trips": [{"launch": 5, "serve": [2], "land": 0}],
    },
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title("LP Solution Visualization", fontsize=14, fontweight="bold")
ax.set_xlabel("X Coordinate (km)")
ax.set_ylabel("Y Coordinate (km)")
ax.grid(True, linestyle="--", alpha=0.3)

# ---- Truck routes with direction arrows ----
for vehicle in vehicle_routes:
    route = vehicle["route"]

    for i in range(len(route) - 1):
        x1, y1 = node_coords[route[i]]
        x2, y2 = node_coords[route[i + 1]]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color="green",
                lw=2,
                alpha=0.7,
            ),
            zorder=1,
        )

# ---- Drone routes with direction arrows ----
for vehicle in vehicle_routes:
    for drone_trip in vehicle["drone_trips"]:
        launch = node_coords[drone_trip["launch"]]
        land = node_coords[drone_trip["land"]]

        for serve_id in drone_trip["serve"]:
            serve = node_coords[serve_id]

            # launch → serve
            ax.annotate(
                "",
                xy=serve,
                xytext=launch,
                arrowprops=dict(
                    arrowstyle="->",
                    color="pink",
                    lw=2,
                    linestyle="--",
                ),
                zorder=2,
            )

            # serve → land
            ax.annotate(
                "",
                xy=land,
                xytext=serve,
                arrowprops=dict(
                    arrowstyle="->",
                    color="pink",
                    lw=2,
                    linestyle="--",
                ),
                zorder=2,
            )

# ---- Draw nodes + labels ----
for node_id, coord in node_coords.items():
    info = node_info[node_id]

    if node_id == 0:
        # Depot
        ax.scatter(coord[0], coord[1], s=220, c="black", edgecolors="black", zorder=10)
        ax.text(
            coord[0],
            coord[1],
            "0",
            color="white",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )
        continue

    color = "blue" if info["type"] == "LINEHAUL" else "red"
    sign = "+" if info["type"] == "LINEHAUL" else "-"

    # Node
    ax.scatter(
        coord[0],
        coord[1],
        s=160,
        c=color,
        edgecolors="black",
        zorder=10,
        alpha=0.9,
    )

    # Node index (inside)
    ax.text(
        coord[0],
        coord[1],
        str(node_id),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=11,
    )

    # Demand (above)
    ax.text(
        coord[0],
        coord[1] + 1.2,
        f"{sign}{abs(info['demand'])}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        fontweight="bold",
        zorder=11,
    )

# ---- Legend ----
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Depot",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Linehaul (+)",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Backhaul (-)",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=10,
    ),
    Line2D([0], [0], color="green", lw=2, label="Truck Route"),
    Line2D([0], [0], color="pink", lw=2, linestyle="--", label="Drone Route"),
]

ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()

# Save
os.makedirs("img", exist_ok=True)
plt.savefig("img/lp.png", dpi=200, bbox_inches="tight")
