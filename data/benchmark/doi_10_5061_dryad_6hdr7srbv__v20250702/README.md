# Routing short-haul trucks under the uncertainties of travel time and service time

Author: Dongbo Peng ([dpeng017@ucr.edu](mailto:dpeng017@ucr.edu))

#### 1.1 Vehicle Routing Problem with Backhauls Benchmark Dataset Description

Folder: Section_3_1_BKS_dataset_and_results

The file "TV_33_instances" contains a total of 33 problem instances in CSV format, original introduced by (Toth & Vigo, 1997), with customer sizes ranging from 21 to 100. Each instance is named according to the number of customers and the proportion of linehaul customers. For example, the file "eil22_50.csv" represents an instance with 22 nodes, where 50% of the customers are linehaul customers. Specifically, in each instance, the columns "node_id", "type", "x", "y", "demand", "Q", "k", "L", and "B" represent the node ID, service type (where -1 denotes the depot node, 0 denotes a linehaul customer, and 1 represents a backhaul customer), the x and y coordinates of the node, the customer demand, vehicle capacity (Q), the vehicle index (k), and the number of linehaul (L) and backhaul (B) customers, respectively.

All the detailed results/solutions are recorded in the Excel file, "Benchmark_TV_results.xlsx." The results table summarizes the performance of the proposed algorithm across multiple problem instances. Each row corresponds to a single instance and includes benchmark comparisons and solution details.

The columns are described as follows:
"Instance": The name of the problem instance, indicating the dataset used (e.g., eil22_50 refers to 22 customers with 50% linehaul customers).
"BKS": The Best Known Solution (BKS) value from the literature or benchmark.
"Best": The best solution value obtained by the proposed algorithm.
"DEV(%)": The percentage deviation of the obtained best solution from the BKS, calculated as DEV(%) = (Our_solution - BKS) / BKS ×100.
t(s): The computational time in seconds required to obtain the best solution.
"BEST_Results": The best route solution(s) found by the algorithm, represented as a list of vehicle routes. Each sublist contains a sequence of node IDs visited by a vehicle, with 0 indicating the depot.

The last row, "Average", shows the average value for each column.

#### 1.2 EVRPBTW-USUT Benchmark Dataset Description

Folder: Section_3_2_EVRPBTW_benchmark

To examine the proposed electric vehicle routing problem with backhauls and time windows under travel time and service time uncertainty (EVRPBTW-USUT), we introduce a new EVRPBTW benchmark dataset based on the well-known EVRPTW benchmark instances introduced by Schneider et al. (Schneider et al., 2014). Our EVRPBTW benchmark dataset incorporates a backhaul strategy for a battery electric vehicle (BEV) fleet, reflecting a practical application in urban logistics and supply chain services. Customers are divided into two groups: linehaul customers (L), who require deliveries, and backhaul customers (L), who require pickups.

We selected 10 instances from the EVRPTW benchmark dataset, where the customers are randomly distributed in the hypothesis scenario. For each instance, similar to (Toth & Vigo, 1997), we generate two EVRPBTW instances based on one EVRPTW dataset, each corresponding to an approximate linehaul percentage of 66% and 75%, respectively. Backhaul customers are grouped in sets of two, three, or four. Other aspects, such as customer locations, time windows, recharging stations, and EV characteristics, remain unchanged. Therefore, for each EVRPTW benchmark dataset, we can generate two new EVRPBTW instances.

For example, the instance named “r201_C25B3” represents a case with 25 customers, where 13 are linehaul customers and 12 are backhaul customers, accounting for approximately 66% backhaul customers. This instance includes 21 recharging stations. In total, 60 instances are generated. The full set of instances can be found in the problem instance files.

In each EVRPBTW instance, it contains the following features:
{'ID': customer ID, 'Type': service type ('D', 'S', 'L', 'B' denote depot, charging station, linehaul customer and backhaul customer, respectively), 'x': x coordinate, 'y': y coordinate, 'demand': customer demand, 'ReadyTime': earliest service start time, 'DueTime': latest service start time, 'ServiceTime': nominal service time}.ßß

For the problem instance parameters, we follow the original problem assumption in (Schneider et al., 2014). For example:
Q: Vehicle fuel tank capacity
C: Vehicle load capacity
r: fuel consumption rate
g: inverse refueling rate
v: average Velocity

#### 1.3 Structure of the benchmark datasets.

Folder: Section_3_3_Real_case_dataset_and_results

The compressed ZIP file "Section_3_1_BKS_dataset_with_results" contains the benchmark dataset obtained from (Toth & Vigo, 1997) along with our solutions. The TV89 instance set comprises 36 problem instances with customer sizes ranging from 21 to 100. Detailed solutions are documented in Excel files.

The new EVRPTWB benchmark dataset contains 60 instances, with 20 instances including 25 customers, 20 instances including 50 customers, and 20 instances including 100 customers. The instances are provided as CSV files. We store the problem instances and detailed solutions in the compressed ZIP file, named "Section_3_2_EVRPBTW_benchmark_instances". The structure of the compressed ZIP file is shown as follow:

\--> Section_3_2_EVRPBTW_benchmark
\--> C25B3
\--> r201_C25B3.csv
\--> r202_C25B3.csv
\--> ...
\--> C25B4
\--> C50B3
\--> C50B4
\--> C100B3
\--> C100B4

The real-world dispatching data is stored in a ZIP file, which includes a CSV file for an instance with 47 customers and an Excel file containing detailed solutions. There are three types of CSV files:
Problem instance: Section3_real_case_data.csv
Distance matrix: real_case_distance_matrix.csv
Time matrix: real_case_time_matrix.csv

This Excel file "Section_3_3_Real_case_solutions.xlsx" provides a comparison of two vehicle routing solutions generated for the same transportation network. Each solution consists of a set of routes originating and ending at a central depot (node 0). The routing results are listed under:

Det_sol: The routing solution obtained from a deterministic model, where travel times, service times, and other parameters are assumed to be known and fixed.
Rob_sol: The routing solution obtained from a robust optimization model, which accounts for uncertainty in travel and service times to enhance solution reliability.
Each solution is represented as a list of routes, with each route defined as a sequence of node IDs. All routes start and end at node 0, which denotes the depot.

References:
Schneider, M., Stenger, A., & Goeke, D. (2014). The electric vehicle-routing problem with time windows and recharging stations. Transportation Science, 48(4), 500–520.
Toth, P., & Vigo, D. (1997). An Exact Algorithm for the Vehicle Routing Problem with Backhauls. Transportation Science, 31(4), 372–385. [https://doi.org/10.1287/trsc.31.4.372](https://doi.org/10.1287/trsc.31.4.372)