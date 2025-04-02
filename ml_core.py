# ml_core.py
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_us_region(city_num):
    if 1 <= city_num <= 10:
        return "West"
    elif 11 <= city_num <= 20:
        return "Midwest"
    elif 21 <= city_num <= 30:
        return "South"
    elif 31 <= city_num <= 40:
        return "Northeast"
    elif 41 <= city_num <= 50:
        return "Southeast"
    else:
        return "Unknown"

def assign_weather_risk(description):
    if description in ["Clear", "Partly Cloudy"]:
        return 1
    elif description in ["Light Rain", "Overcast", "Mild Snow"]:
        return 2
    elif description in ["Moderate Rain", "Moderate Snow", "Windy"]:
        return 3
    elif description in ["Heavy Snow", "Storm", "Fog", "Torrential Rain", "Thunderstorm"]:
        return 5
    else:
        return 4

def classify_traffic(no_of_vehicles, accident):
    if accident:
        return 2.0
    elif no_of_vehicles >= 100:
        return 1.5
    elif no_of_vehicles >= 50:
        return 1.25
    else:
        return 1.0

def load_data():
    base = 'data/'
    df_routes = pd.read_csv(base + "routes_table.csv", index_col=0)
    df_city_weather = pd.read_csv(base + "city_weather.csv")
    df_drivers = pd.read_csv(base + "drivers_table.csv")
    df_routes_weather = pd.read_csv(base + "routes_weather.csv")
    df_traffic = pd.read_csv(base + "traffic_table.csv")
    df_trucks = pd.read_csv(base + "trucks_table.csv")
    df_truck_schedule = pd.read_csv(base + "truck_schedule_table.csv")
    return df_routes, df_routes_weather, df_traffic

def run_pipeline(region_id, start_city, destination_city, optimize_for):
    region_mapping = {1: "West", 2: "Midwest", 3: "South", 4: "Northeast", 5: "Southeast"}
    region_name = region_mapping[region_id]

    df_routes, df_routes_weather, df_traffic = load_data()

    df_routes["origin_region"] = df_routes["origin_id"].apply(lambda x: get_us_region(int(x[1:])))
    df_routes["destination_region"] = df_routes["destination_id"].apply(lambda x: get_us_region(int(x[1:])))
    df_routes_weather["weather"] = df_routes_weather["description"].apply(assign_weather_risk)
    df_traffic["traffic_severity"] = df_traffic.apply(
        lambda row: classify_traffic(row["no_of_vehicles"], row["accident"]), axis=1
    )
    df_traffic = df_traffic.groupby("route_id", as_index=False)["traffic_severity"].max()

    df_routes = df_routes.merge(df_traffic, on="route_id", how="left")
    df_routes["traffic_severity"].fillna(1.0, inplace=True)

    regional_routes = df_routes.merge(df_routes_weather, on="route_id", how="left")
    regional_routes = regional_routes[
        (regional_routes["origin_region"] == region_name) &
        (regional_routes["destination_region"] == region_name)
    ]

    def create_graph(filtered_routes):
        G = nx.Graph()
        for _, row in filtered_routes.iterrows():
            origin = row["origin_id"]
            destination = row["destination_id"]
            distance = row.get("distance", float("inf"))
            time_val = row.get("average_hours", float("inf"))
            weather_risk = row.get("weather", 3)
            traffic_severity = row.get("traffic_severity", 1.0)
            adjusted_time = time_val * traffic_severity if optimize_for == "time" else time_val

            if distance != float("inf") and time_val != float("inf"):
                G.add_edge(origin, destination, distance=distance, time=adjusted_time, weather_risk=weather_risk)
        return G

    G = create_graph(regional_routes)

    training_data = []
    for origin, destination, data in G.edges(data=True):
        training_data.append([origin, destination, data['distance'], data['time'], data['weather_risk']])
        training_data.append([destination, origin, data['distance'], data['time'], data['weather_risk']])

    df_train = pd.DataFrame(training_data, columns=['origin', 'destination', 'distance', 'time', 'weather_risk'])
    X = pd.get_dummies(df_train[['origin', 'distance', 'time', 'weather_risk']])
    y = df_train['destination']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cities_in_region = list(range((region_id - 1) * 10 + 1, region_id * 10 + 1))
    cities_in_region_ids = [f"C{num}" for num in cities_in_region]
    intermediate_cities = set(cities_in_region_ids) - {start_city, destination_city}
    current_city = start_city
    final_path = [start_city]
    unvisited = intermediate_cities.copy()

    while unvisited:
        next_city = None
        best_value = float("inf")
        for candidate in unvisited:
            try:
                value = nx.shortest_path_length(G, source=current_city, target=candidate, weight=optimize_for)
            except:
                continue
            if value < best_value:
                best_value = value
                next_city = candidate
        if next_city is None:
            break
        path_segment = nx.shortest_path(G, source=current_city, target=next_city, weight=optimize_for)
        final_path += path_segment[1:]
        unvisited.remove(next_city)
        current_city = next_city

    try:
        path_segment = nx.shortest_path(G, source=current_city, target=destination_city, weight=optimize_for)
        final_path += path_segment[1:]
    except:
        pass

    total_distance = 0
    total_time = 0
    for i in range(len(final_path) - 1):
        edge_data = G.get_edge_data(final_path[i], final_path[i + 1])
        total_distance += edge_data.get("distance", 0)
        total_time += edge_data.get("time", 0)

    return {
        "accuracy": round(accuracy * 100, 2),
        "path": final_path,
        "total_distance": round(total_distance, 2),
        "total_time": round(total_time, 2)
    }


