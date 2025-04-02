import pandas as pd
import networkx as nx
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load datasets
df_routes = pd.read_csv("routes_table.csv", index_col=0)
df_routes_weather = pd.read_csv("routes_weather.csv")

# Assign weather risk
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

df_routes_weather["weather_risk"] = df_routes_weather["description"].apply(assign_weather_risk)
df_routes = df_routes.merge(df_routes_weather[["route_id", "weather_risk"]], on="route_id", how="left")

# Region city mapping
region_mapping = {
    "West": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
    "Midwest": ["C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20"],
    "South": ["C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30"],
    "Northeast": ["C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40"],
    "Southeast": ["C41", "C42", "C43", "C44", "C45", "C46", "C47", "C48", "C49", "C50"]
}

# Coordinates dictionary (shortened here for space — keep yours)
city_coordinates = {
   # WEST (C1 - C10)
    "C1": [34.0522, -118.2437], "C2": [36.1699, -115.1398], "C3": [37.7749, -122.4194], 
    "C4": [39.7392, -104.9903], "C5": [32.7157, -117.1611], "C6": [45.5051, -122.6750], 
    "C7": [47.6062, -122.3321], "C8": [35.0844, -106.6504], "C9": [40.7608, -111.8910], 
    "C10": [33.4484, -112.0740],

    # MIDWEST (C11 - C20)
    "C11": [41.2565, -95.9345], "C12": [39.0997, -94.5786], "C13": [38.2527, -85.7585], 
    "C14": [43.0389, -87.9065], "C15": [42.3314, -83.0458], "C16": [44.9778, -93.2650], 
    "C17": [40.4406, -79.9959], "C18": [39.7684, -86.1581], "C19": [35.1495, -90.0490], 
    "C20": [30.2672, -97.7431],

    # SOUTH (C21 - C30)
    "C21": [29.7604, -95.3698],  # Houston, TX
    "C22": [35.2271, -80.8431],  # Charlotte, NC
    "C23": [36.1627, -86.7816],  # Nashville, TN
    "C24": [30.3322, -81.6557],  # Jacksonville, FL
    "C25": [25.7617, -80.1918],  # Miami, FL
    "C26": [32.0835, -81.0998],  # Savannah, GA
    "C27": [27.9506, -82.4572],  # Tampa, FL
    "C28": [29.9511, -90.0715],  # New Orleans, LA
    "C29": [34.7304, -86.5861],  # Huntsville, AL
    "C30": [37.5407, -77.4360],  # Richmond, VA

    # NORTHEAST (C31 - C40)
    "C31": [42.3601, -71.0589],  # Boston, MA
    "C32": [39.2904, -76.6122],  # Baltimore, MD
    "C33": [40.7128, -74.0060],  # New York, NY
    "C34": [38.9072, -77.0369],  # Washington, DC
    "C35": [40.7357, -74.1724],  # Newark, NJ
    "C36": [36.8508, -76.2859],  # Norfolk, VA
    "C37": [44.9778, -93.2650],  # St. Paul, MN
    "C38": [33.7490, -84.3880],  # Atlanta, GA
    "C39": [41.2033, -77.1945],  # Harrisburg, PA
    "C40": [40.2732, -76.8867],  # Scranton, PA

    # SOUTHEAST (C41 - C50)
    "C41": [33.7490, -84.3880],  # Atlanta, GA
    "C42": [35.4676, -97.5164],  # Oklahoma City, OK
    "C43": [36.1627, -86.7816],  # Nashville, TN
    "C44": [32.7767, -96.7970],  # Dallas, TX
    "C45": [27.9506, -82.4572],  # Tampa, FL
    "C46": [30.2672, -97.7431],  # Austin, TX
    "C47": [29.4241, -98.4936],  # San Antonio, TX
    "C48": [39.9526, -75.1652],  # Philadelphia, PA
    "C49": [40.7608, -111.8910], # Salt Lake City, UT
    "C50": [42.3314, -83.0458]   # Detroit, MI
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_route', methods=['POST'])
def find_route():
    data = request.json
    region = data['region']
    start = data['start_city']
    end = data['destination_city']
    optimize_for = data['optimize_for']

    # Filter region
    cities = region_mapping[region]
    df = df_routes[(df_routes["origin_id"].isin(cities)) & (df_routes["destination_id"].isin(cities))]

    # Graph and training data
    G = nx.Graph()
    training_data = []
    for _, row in df.iterrows():
        origin = row["origin_id"]
        dest = row["destination_id"]
        dist = row.get("distance", float("inf"))
        t = row.get("average_hours", float("inf"))
        w = row.get("weather_risk", 3)
        G.add_edge(origin, dest, distance=dist, time=t, weather_risk=w)
        training_data.append([origin, dest, dist, t, w])
        training_data.append([dest, origin, dist, t, w])  # bidirectional

    # ML setup
    df_train = pd.DataFrame(training_data, columns=["origin", "destination", "distance", "time", "weather_risk"])
    X = pd.get_dummies(df_train[["origin", "distance", "time", "weather_risk"]])
    y = df_train["destination"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Greedy TSP-style routing
    unvisited = set(cities) - {start, end}
    current = start
    path = [start]

    while unvisited:
        next_city = None
        best_cost = float("inf")
        for city in unvisited:
            try:
                cost = nx.shortest_path_length(G, current, city, weight=optimize_for)
                if cost < best_cost:
                    best_cost = cost
                    next_city = city
            except:
                continue
        if not next_city:
            break
        segment = nx.shortest_path(G, current, next_city, weight=optimize_for)
        path += segment[1:]
        unvisited.remove(next_city)
        current = next_city

    try:
        segment = nx.shortest_path(G, current, end, weight=optimize_for)
        path += segment[1:]
    except:
        return jsonify({"error": f"No path from {current} to {end}."})

    # Metrics
    total_distance = sum(G[u][v]["distance"] for u, v in zip(path[:-1], path[1:]))
    total_time = sum(G[u][v]["time"] for u, v in zip(path[:-1], path[1:]))
    avg_weather = sum(G[u][v]["weather_risk"] for u, v in zip(path[:-1], path[1:])) / (len(path)-1)

    return jsonify({
        "optimal_path": path,
        "total_distance": round(total_distance, 2),
        "total_time": round(total_time, 2),
        "avg_weather_risk": round(avg_weather, 2),
        "coordinates": {city: city_coordinates.get(city, [0, 0]) for city in path},
        "ml_accuracy": round(accuracy * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)