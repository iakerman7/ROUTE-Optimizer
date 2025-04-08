import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import itertools
import warnings
warnings.filterwarnings('ignore')

class RouteOptimizationSystem:
    """
    Advanced routing system that combines multiple data sources and ML
    to optimize routes based on distance, time, and weather conditions.
    """
    
    def __init__(self):
        self.routes_df = None
        self.weather_df = None
        self.traffic_df = None
        self.drivers_df = None
        self.trucks_df = None
        self.schedule_df = None
        self.city_weather_df = None
        self.travel_time_model = None
        self.feature_preprocessor = None
        self.G = None
        self.region_mapping = {
            1: "West", 2: "Midwest", 3: "South", 4: "Northeast", 5: "Southeast"
        }
        # Store all different graph versions
        self.graphs = {
            "time": None,
            "distance": None,
            "weather": None,
            "combined": None
        }
        self.last_region = None
        
    def load_data(self, routes_path, weather_path, traffic_path, 
                 drivers_path=None, trucks_path=None, schedule_path=None,
                 city_weather_path=None):
        """Load all required datasets"""
        
        print("Loading data...")
        # Core datasets
        self.routes_df = pd.read_csv(routes_path)
        self.weather_df = pd.read_csv(weather_path)
        self.traffic_df = pd.read_csv(traffic_path)
        
        # Optional datasets for enhanced modeling
        if drivers_path:
            self.drivers_df = pd.read_csv(drivers_path)
        if trucks_path:
            self.trucks_df = pd.read_csv(trucks_path)
        if schedule_path:
            self.schedule_df = pd.read_csv(schedule_path)
        if city_weather_path:
            self.city_weather_df = pd.read_csv(city_weather_path)
            
        print("Data loaded successfully.")
        
    def preprocess_data(self):
        """Preprocess and integrate all datasets"""
        
        print("Preprocessing data...")
        
        # Add region information to routes
        self.routes_df["origin_region"] = self.routes_df["origin_id"].apply(self._get_region)
        self.routes_df["destination_region"] = self.routes_df["destination_id"].apply(self._get_region)
        
        # Process weather data
        if 'description' in self.weather_df.columns:
            self.weather_df["weather_risk"] = self.weather_df["description"].apply(self._assign_weather_risk)
        
        # Aggregate weather risks by route
        route_weather = self.weather_df.groupby('route_id')['weather_risk'].agg(
            ['mean', 'max', 'std']).reset_index()
        route_weather.columns = ['route_id', 'avg_weather_risk', 'max_weather_risk', 'weather_variability']
        
        # Process traffic data
        self.traffic_df["traffic_severity"] = self.traffic_df.apply(
            lambda row: self._classify_traffic(row["no_of_vehicles"], row["accident"]), axis=1
        )
        
        # Aggregate traffic by route
        route_traffic = self.traffic_df.groupby('route_id')['traffic_severity'].agg(
            ['mean', 'max', 'std']).reset_index()
        route_traffic.columns = ['route_id', 'avg_traffic', 'max_traffic', 'traffic_variability']
        
        # Integrate with main routes table
        self.processed_routes = self.routes_df.merge(route_weather, on='route_id', how='left')
        self.processed_routes = self.processed_routes.merge(route_traffic, on='route_id', how='left')
        
        # Fill missing values with reasonable defaults
        self.processed_routes['avg_weather_risk'].fillna(3, inplace=True)
        self.processed_routes['max_weather_risk'].fillna(3, inplace=True)
        self.processed_routes['weather_variability'].fillna(0, inplace=True)
        self.processed_routes['avg_traffic'].fillna(1, inplace=True)
        self.processed_routes['max_traffic'].fillna(1, inplace=True)
        self.processed_routes['traffic_variability'].fillna(0, inplace=True)
        
        # Add seasonal features if date information is available
        if 'Date' in self.weather_df.columns:
            # Extract season information
            self.weather_df['date_obj'] = pd.to_datetime(self.weather_df['Date'], dayfirst=True)
            self.weather_df['month'] = self.weather_df['date_obj'].dt.month
            self.weather_df['season'] = self.weather_df['month'].apply(self._get_season)
            
            # Aggregate seasonal weather risks
            season_risks = self.weather_df.groupby(['route_id', 'season'])['weather_risk'].mean().reset_index()
            season_risks_pivot = season_risks.pivot(index='route_id', columns='season', values='weather_risk').reset_index()
            expected_columns = ['route_id', 'spring_risk', 'summer_risk', 'fall_risk', 'winter_risk']
            if season_risks_pivot.shape[1] == len(expected_columns):
                season_risks_pivot.columns = expected_columns 
            else:
                    print("⚠️ Warning: Unexpected number of columns in season_risks_pivot") 
                    print("  Found columns:", list(season_risks_pivot.columns)) 
                    season_risks_pivot.columns = season_risks_pivot.columns  # fallback

            
            # Merge seasonal features
            self.processed_routes = self.processed_routes.merge(season_risks_pivot, on='route_id', how='left')
            
            # Fill missing seasonal values
            for season in ['spring_risk', 'summer_risk', 'fall_risk', 'winter_risk']:
                if season in self.processed_routes.columns:
                    self.processed_routes[season].fillna(3, inplace=True)
                else: 
                    print(f"⚠️ Column '{season}' not found in processed_routes — skipping.")
        
        # Create additional derived features
        # Calculate speed (miles per hour)
        self.processed_routes['speed'] = self.processed_routes['distance'] / self.processed_routes['average_hours']
        self.processed_routes['speed'].fillna(40, inplace=True)  # Default 40 mph
        self.processed_routes['speed'].replace([np.inf, -np.inf], 40, inplace=True)
        
        # Create road type factor (deterministic but varies by route) to help differentiate time vs distance
        self.processed_routes['road_quality'] = self.processed_routes.apply(
            lambda row: self._calculate_road_quality(row['route_id'], row['distance']), axis=1
        )
        
        print("Data preprocessing complete.")
        print(f"Processed routes shape: {self.processed_routes.shape}")
        
    def build_model(self):
        """Build and train ML model for travel time prediction"""
        
        print("Building travel time prediction model...")
        
        # Features for modeling
        features = [
            'distance', 'avg_weather_risk', 'max_weather_risk', 'weather_variability',
            'avg_traffic', 'max_traffic', 'traffic_variability',
            'origin_region', 'destination_region'
        ]
        
        # Add seasonal features if available
        if 'spring_risk' in self.processed_routes.columns:
            features.extend(['spring_risk', 'summer_risk', 'fall_risk', 'winter_risk'])
        
        # Target: travel time
        target = 'average_hours'
        
        # Prepare the data
        X = self.processed_routes[features]
        y = self.processed_routes[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build preprocessing pipeline
        categorical_features = ['origin_region', 'destination_region']
        numerical_features = [f for f in features if f not in categorical_features]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Full pipeline with model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.1]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.travel_time_model = grid_search.best_estimator_
        
        # Evaluate model
        predictions = self.travel_time_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model training complete.")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Mean Absolute Error: {mae:.2f} hours")
        print(f"R² Score: {r2:.4f}")
        
        # Save the preprocessor for later use
        self.feature_preprocessor = self.travel_time_model.named_steps['preprocessor']
        
        return self.travel_time_model
    
    def find_weather_optimized_route(self, region_id, start_city, end_city):
        """
        Find a route specifically optimized for minimum weather risk.
        This is a dedicated implementation for weather optimization that bypasses
        the general TSP approach to focus solely on minimizing weather risk.
        
        Parameters:
        region_id (int): Region ID (1-5)
        start_city (str): Starting city ID
        end_city (str): Ending city ID
        
        Returns:
        dict: Path and metrics
        """
        print(f"\nFinding weather-optimized route from {start_city} to {end_city} in region {region_id}...")
        
        # Filter routes by region
        region_name = self.region_mapping[region_id]
        regional_routes = self.processed_routes[
            (self.processed_routes["origin_region"] == region_name) &
            (self.processed_routes["destination_region"] == region_name)
        ]
        
        # Create a graph specifically for weather optimization
        G_weather = nx.Graph()
        
        # Add all cities in the region as nodes
        cities_in_region = [f"C{n}" for n in range((region_id - 1) * 10 + 1, region_id * 10 + 1)]
        for city in cities_in_region:
            G_weather.add_node(city)
        
        # Add edges with weights based ONLY on weather risk
        for _, row in regional_routes.iterrows():
            origin = row["origin_id"]
            destination = row["destination_id"]
            
            if origin in G_weather.nodes and destination in G_weather.nodes:
                # Use only weather risk as weight - directly use the risk value
                # Lower risk = lower weight = preferred path
                weather_weight = row["avg_weather_risk"]
                
                G_weather.add_edge(
                    origin, 
                    destination, 
                    weight=weather_weight,
                    distance=row["distance"],
                    time=row["average_hours"],
                    raw_time=row["average_hours"],
                    weather_risk=row["avg_weather_risk"],
                    traffic=row["avg_traffic"],
                    route_id=row["route_id"]
                )
        
        # Check if all cities are connected
        components = list(nx.connected_components(G_weather))
        if len(components) > 1:
            print(f"Warning: Graph has {len(components)} disconnected components.")
            # Find the component with start_city
            main_component = None
            for component in components:
                if start_city in component:
                    main_component = component
                    break
            
            if main_component:
                # Keep only cities in this component
                cities_in_region = [city for city in cities_in_region if city in main_component]
                print(f"Using only {len(cities_in_region)} cities that are connected.")
            else:
                print(f"Error: Start city {start_city} not found in any component.")
                return {
                    "path": [start_city],
                    "total_distance": 0,
                    "total_time": 0,
                    "avg_weather_risk": 0,
                    "visited_cities": 1,
                    "expected_cities": len(cities_in_region),
                    "optimization": "weather",
                    "error": "Start city not connected"
                }
        
        # Make sure start and end cities are in the graph
        if start_city not in G_weather or end_city not in G_weather:
            print(f"Error: Start or end city not in graph.")
            return {
                "path": [start_city],
                "total_distance": 0,
                "total_time": 0,
                "avg_weather_risk": 0,
                "visited_cities": 1,
                "expected_cities": len(cities_in_region),
                "optimization": "weather",
                "error": "Start or end city not in graph"
            }
        
        # Create a complete graph for the TSP with weights based on paths that minimize weather risk
        tsp_graph = nx.Graph()
        
        # Add all cities as nodes
        for city in cities_in_region:
            tsp_graph.add_node(city)
        
        # Calculate shortest paths between all pairs of cities using weather risk as weight
        for i, city1 in enumerate(cities_in_region):
            for city2 in cities_in_region[i+1:]:
                if city1 != city2:
                    try:
                        # Find path with lowest weather risk
                        path = nx.shortest_path(G_weather, source=city1, target=city2, weight='weight')
                        
                        # Calculate the total weather risk along this path
                        total_risk = 0
                        total_distance = 0
                        total_time = 0
                        
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j+1]
                            edge_data = G_weather.get_edge_data(u, v)
                            total_risk += edge_data['weather_risk']
                            total_distance += edge_data['distance']
                            total_time += edge_data['time']
                        
                        # Average risk along the path
                        avg_risk = total_risk / (len(path) - 1) if len(path) > 1 else 0
                        
                        # Add edge with weight = average weather risk and store the path
                        tsp_graph.add_edge(
                            city1, city2, 
                            weight=avg_risk,  # Use average risk as the weight
                            path=path,
                            distance=total_distance,
                            time=total_time
                        )
                    except nx.NetworkXNoPath:
                        continue
        
        # Solve the TSP to visit all cities once with minimum weather risk
        current_city = start_city
        final_path = [start_city]
        unvisited = set(cities_in_region) - {start_city}
        
        # If end_city is specified and different from start_city,
        # handle it specially
        if end_city != start_city:
            # Remove end_city from unvisited as we'll add it at the end
            if end_city in unvisited:
                unvisited.remove(end_city)
        
        # Visit all intermediate cities
        while unvisited:
            next_city = None
            best_risk = float('inf')
            
            for candidate in unvisited:
                if tsp_graph.has_edge(current_city, candidate):
                    risk = tsp_graph[current_city][candidate]['weight']
                    if risk < best_risk:
                        best_risk = risk
                        next_city = candidate
            
            if next_city is None:
                print(f"Warning: No path found from {current_city} to any unvisited city.")
                break
            
            final_path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Add end_city if it's not already the last city
        if end_city != start_city and end_city not in final_path:
            final_path.append(end_city)
        
        # Calculate metrics for the final path
        total_distance = 0
        total_time = 0
        total_weather_risk = 0
        
        # Expand the path to include all intermediate cities
        expanded_path = []
        
        for i in range(len(final_path) - 1):
            source = final_path[i]
            target = final_path[i + 1]
            
            if tsp_graph.has_edge(source, target):
                # Get the detailed path between these cities
                segment_path = tsp_graph[source][target]['path']
                
                # Add the first city of this segment
                if not expanded_path or expanded_path[-1] != segment_path[0]:
                    expanded_path.append(segment_path[0])
                
                # Add the rest of the segment
                expanded_path.extend(segment_path[1:])
                
                # Add metrics
                total_distance += tsp_graph[source][target]['distance']
                total_time += tsp_graph[source][target]['time']
                
                # Calculate weather risk for this segment
                for j in range(len(segment_path) - 1):
                    u, v = segment_path[j], segment_path[j+1]
                    if G_weather.has_edge(u, v):
                        edge_data = G_weather.get_edge_data(u, v)
                        total_weather_risk += edge_data['weather_risk']
            else:
                print(f"Warning: No edge between {source} and {target}")
        
        # Calculate average weather risk
        avg_weather_risk = total_weather_risk / (len(expanded_path) - 1) if len(expanded_path) > 1 else 0
        
        # Return results
        results = {
            "path": final_path,  # Simplified path with unique cities
            "detailed_path": expanded_path,  # Full path with all intermediate cities
            "total_distance": round(total_distance, 2),
            "total_time": round(total_time, 2),
            "avg_weather_risk": round(avg_weather_risk, 2),
            "visited_cities": len(set(final_path)),
            "expected_cities": len(cities_in_region),
            "optimization": "weather"
        }
        
        print(f"Weather-optimized route planning complete.")
        print(f"Total distance: {results['total_distance']} miles")
        print(f"Total time: {results['total_time']} hours")
        print(f"Average weather risk: {results['avg_weather_risk']} (1-5 scale)")
        print(f"Visited {results['visited_cities']} out of {results['expected_cities']} cities")
        
        return results
    
    def find_time_optimized_route(self, region_id, start_city, end_city):
        """
        Find a route specifically optimized for minimum travel time.
        This version uses multiple techniques to find the truly fastest route.
        
        Parameters:
        region_id (int): Region ID (1-5)
        start_city (str): Starting city ID
        end_city (str): Ending city ID
        
        Returns:
        dict: Path and metrics
        """
        print(f"\nFinding time-optimized route from {start_city} to {end_city} in region {region_id}...")
        
        # Filter routes by region
        region_name = self.region_mapping[region_id]
        regional_routes = self.processed_routes[
            (self.processed_routes["origin_region"] == region_name) &
            (self.processed_routes["destination_region"] == region_name)
        ]
        
        # Create a graph specifically for time optimization
        G_time = nx.Graph()
        
        # Add all cities in the region as nodes
        cities_in_region = [f"C{n}" for n in range((region_id - 1) * 10 + 1, region_id * 10 + 1)]
        for city in cities_in_region:
            G_time.add_node(city)
        
        # Add edges with weights based ONLY on pure travel time
        for _, row in regional_routes.iterrows():
            origin = row["origin_id"]
            destination = row["destination_id"]
            
            if origin in G_time.nodes and destination in G_time.nodes:
                # Use pure travel time as the weight - no adjustments
                travel_time = row["average_hours"]
                
                G_time.add_edge(
                    origin, 
                    destination, 
                    weight=travel_time,  # This is the key weight for optimization
                    distance=row["distance"],
                    time=row["average_hours"],
                    raw_time=row["average_hours"],
                    weather_risk=row["avg_weather_risk"],
                    traffic=row["avg_traffic"],
                    route_id=row["route_id"]
                )
        
        # Check if cities are missing from the graph
        missing_cities = [city for city in cities_in_region if city not in G_time.nodes]
        if missing_cities:
            print(f"Warning: Cities {missing_cities} are not in the graph.")
            cities_in_region = [city for city in cities_in_region if city in G_time.nodes]
        
        # Check if all cities are connected
        if not nx.is_connected(G_time):
            print("Warning: Not all cities are connected in the region.")
            # Get the connected component containing the start city
            for component in nx.connected_components(G_time):
                if start_city in component:
                    cities_in_region = [city for city in cities_in_region if city in component]
                    break
        
        # Make sure start and end cities are in the graph
        if start_city not in G_time or end_city not in G_time:
            print(f"Error: Start or end city not in graph.")
            return {
                "path": [start_city],
                "total_distance": 0,
                "total_time": 0,
                "avg_weather_risk": 0,
                "visited_cities": 1,
                "expected_cities": len(cities_in_region),
                "optimization": "time",
                "error": "Start or end city not in graph"
            }
        
        # STRATEGY 1: Try a variety of different algorithms and select the best result
        best_path = None
        best_time = float('inf')
        best_distance = 0
        best_weather_risk = 0
        
        # 1. Brute force for small problems (≤ 8 cities)
        if len(cities_in_region) <= 8:
            # Intermediate cities (excluding start and end)
            intermediate_cities = [c for c in cities_in_region if c != start_city and c != end_city]
            
            # Try all permutations of intermediate cities
            for perm in itertools.permutations(intermediate_cities):
                # Create the full path: start -> intermediate cities -> end
                path = [start_city] + list(perm) + [end_city]
                
                # Calculate metrics
                time, distance, weather = self._calculate_path_metrics(G_time, path)
                
                if time < best_time:
                    best_time = time
                    best_path = path
                    best_distance = distance
                    best_weather_risk = weather
            
            if best_path:
                print(f"Brute force found a path with time: {best_time}")
        
        # 2. Try greedy nearest neighbor (always choose the next closest city by time)
        greedy_path, greedy_time, greedy_distance, greedy_weather = self._find_greedy_time_route(
            G_time, cities_in_region, start_city, end_city)
        
        if greedy_path and (best_path is None or greedy_time < best_time):
            best_path = greedy_path
            best_time = greedy_time
            best_distance = greedy_distance
            best_weather_risk = greedy_weather
            print(f"Greedy approach found a path with time: {best_time}")
        
        # If no valid path was found, use standard TSP approach as fallback
        if best_path is None:
            print("No valid path found. Using standard TSP approach.")
            # Create a more connected graph by allowing multi-hop paths
            tsp_graph = nx.Graph()
            for city in cities_in_region:
                tsp_graph.add_node(city)
            
            # Add edges for all city pairs where a path exists in the original graph
            for i, city1 in enumerate(cities_in_region):
                for city2 in cities_in_region[i+1:]:
                    if city1 != city2:
                        try:
                            path_length = nx.shortest_path_length(
                                G_time, source=city1, target=city2, weight='weight')
                            path = nx.shortest_path(
                                G_time, source=city1, target=city2, weight='weight')
                            tsp_graph.add_edge(city1, city2, weight=path_length, path=path)
                        except nx.NetworkXNoPath:
                            pass
            
            # Standard nearest neighbor TSP with start and end cities fixed
            # Start with start_city
            current_city = start_city
            tsp_path = [start_city]
            unvisited = set(cities_in_region) - {start_city}
            
            # Remove end_city if different from start_city (will add at end)
            if end_city != start_city and end_city in unvisited:
                unvisited.remove(end_city)
            
            # Visit all intermediate cities
            while unvisited:
                next_city = None
                best_time_to_next = float('inf')
                
                for candidate in unvisited:
                    if tsp_graph.has_edge(current_city, candidate):
                        edge_time = tsp_graph[current_city][candidate]['weight']
                        if edge_time < best_time_to_next:
                            best_time_to_next = edge_time
                            next_city = candidate
                
                if next_city is None:
                    print(f"Warning: No path found from {current_city} to any unvisited city.")
                    break
                
                tsp_path.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
            
            # Add end city if necessary
            if end_city != start_city and end_city not in tsp_path:
                tsp_path.append(end_city)
            
            # Calculate metrics
            best_path = tsp_path
            best_time, best_distance, best_weather_risk = self._calculate_path_metrics(G_time, tsp_path)
        
        # Calculate average weather risk
        avg_weather_risk = best_weather_risk / (len(best_path) - 1) if len(best_path) > 1 else 0
        
        results = {
            "path": best_path,
            "detailed_path": best_path,  # For consistency with other methods
            "total_distance": round(best_distance, 2),
            "total_time": round(best_time, 2),
            "avg_weather_risk": round(avg_weather_risk, 2),
            "visited_cities": len(best_path),
            "expected_cities": len(cities_in_region),
            "optimization": "time"
        }
        
        print(f"Time-optimized route planning complete.")
        print(f"Total distance: {results['total_distance']} miles")
        print(f"Total time: {results['total_time']} hours")
        print(f"Average weather risk: {results['avg_weather_risk']} (1-5 scale)")
        print(f"Visited {results['visited_cities']} out of {results['expected_cities']} cities")
        
        return results
    
    def _find_greedy_time_route(self, G, cities_in_region, start_city, end_city):
        """
        Find a time-optimized route using a greedy nearest-neighbor approach.
        Used as a fallback when brute force is not practical.
        """
        current_city = start_city
        path = [start_city]
        unvisited = set(cities_in_region) - {start_city, end_city}
        
        total_time = 0
        total_distance = 0
        total_weather_risk = 0
        
        # Visit all intermediate cities
        while unvisited:
            next_city = None
            best_time = float('inf')
            
            for candidate in unvisited:
                if G.has_edge(current_city, candidate):
                    time = G[current_city][candidate]['time']
                    if time < best_time:
                        best_time = time
                        next_city = candidate
            
            if next_city is None:
                print(f"Warning: No path found from {current_city} to any unvisited city.")
                break
            
            path.append(next_city)
            
            # Update metrics
            edge_data = G.get_edge_data(current_city, next_city)
            total_time += edge_data['time']
            total_distance += edge_data['distance']
            total_weather_risk += edge_data['weather_risk']
            
            unvisited.remove(next_city)
            current_city = next_city
        
        # Add end city
        if G.has_edge(current_city, end_city):
            path.append(end_city)
            
            edge_data = G.get_edge_data(current_city, end_city)
            total_time += edge_data['time']
            total_distance += edge_data['distance']
            total_weather_risk += edge_data['weather_risk']
        else:
            print(f"Warning: No path from {current_city} to end city {end_city}.")
        
        return path, total_time, total_distance, total_weather_risk
    
    def _calculate_path_metrics(self, G, path):
        """
        Calculate total time, distance, and weather risk for a path.
        """
        total_time = 0
        total_distance = 0
        total_weather_risk = 0
        
        # Check if the path has at least 2 cities
        if len(path) < 2:
            return 0, 0, 0
        
        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            
            # Check if there's a direct edge
            if G.has_edge(city1, city2):
                edge_data = G.get_edge_data(city1, city2)
                total_time += edge_data['time']
                total_distance += edge_data['distance']
                total_weather_risk += edge_data['weather_risk']
            else:
                # If no direct edge, find the shortest path
                try:
                    shortest_path = nx.shortest_path(G, city1, city2, weight='weight')
                    
                    # Sum up the metrics along this path
                    for j in range(len(shortest_path) - 1):
                        u, v = shortest_path[j], shortest_path[j+1]
                        edge_data = G.get_edge_data(u, v)
                        total_time += edge_data['time']
                        total_distance += edge_data['distance']
                        total_weather_risk += edge_data['weather_risk']
                except nx.NetworkXNoPath:
                    # If no path exists, return infinity
                    print(f"No path found between {city1} and {city2}")
                    return float('inf'), 0, 0
        
        return total_time, total_distance, total_weather_risk
    
    def build_graph(self, region_id, optimize_for):
        """
        Build a NetworkX graph for route optimization.
        
        Parameters:
        region_id (int): Region ID to filter routes
        optimize_for (str): 'time', 'distance', 'weather', or 'combined'
        
        Returns:
        nx.Graph: The constructed graph
        """
        print(f"Building optimization graph for {optimize_for} optimization...")
        
        # Filter routes by region
        region_name = self.region_mapping[region_id]
        regional_routes = self.processed_routes[
            (self.processed_routes["origin_region"] == region_name) &
            (self.processed_routes["destination_region"] == region_name)
        ]
        
        # Create graph
        G = nx.Graph()
        
        # Get statistics for normalization
        max_dist = regional_routes["distance"].max()
        max_time = regional_routes["average_hours"].max()
        
        # Different weight calculations for each optimization type
        for _, row in regional_routes.iterrows():
            origin = row["origin_id"]
            destination = row["destination_id"]
            
            # Base attributes that don't change
            attrs = {
                'distance': row["distance"],
                'time': row["average_hours"],
                'raw_time': row["average_hours"],
                'weather_risk': row["avg_weather_risk"],
                'traffic': row["avg_traffic"],
                'route_id': row["route_id"],
                'speed': row["speed"],
                'road_quality': row["road_quality"]
            }
            
            # Calculate edge weight based on optimization goal
            if optimize_for == "time":
                # TIME OPTIMIZATION
                # Use pure travel time with minimal traffic adjustment
                traffic_factor = 1.0 + (row["avg_traffic"] - 1.0) * 0.3
                attrs['weight'] = row["average_hours"] * traffic_factor
                
                # Store original optimization factors for reference
                attrs['time_weight'] = attrs['weight']
                attrs['distance_weight'] = row["distance"]
                attrs['weather_weight'] = row["avg_weather_risk"]
            
            elif optimize_for == "distance":
                # DISTANCE OPTIMIZATION
                # Simply use pure distance
                attrs['weight'] = row["distance"]
                
                # Store original optimization factors for reference
                attrs['time_weight'] = row["average_hours"] * row["avg_traffic"]
                attrs['distance_weight'] = attrs['weight']
                attrs['weather_weight'] = row["avg_weather_risk"]
            
            elif optimize_for == "weather":
                # WEATHER OPTIMIZATION
                # Use direct weather risk as the weight
                attrs['weight'] = row["avg_weather_risk"]
                
                # Store original optimization factors for reference
                attrs['time_weight'] = row["average_hours"] * row["avg_traffic"]
                attrs['distance_weight'] = row["distance"]
                attrs['weather_weight'] = attrs['weight']
            
            elif optimize_for == "combined":
                # COMBINED OPTIMIZATION
                # Balance all three factors
                # Normalize each component
                norm_dist = row["distance"] / max_dist if max_dist > 0 else 0
                norm_time = row["average_hours"] / max_time if max_time > 0 else 0
                norm_weather = row["avg_weather_risk"] / 5.0  # Simple scaling for 1-5 range
                
                # Combined weight with equal weights
                attrs['weight'] = (
                    0.33 * norm_dist +
                    0.33 * norm_time * row["avg_traffic"] +
                    0.34 * norm_weather
                )
                
                # Store original optimization factors for reference
                attrs['time_weight'] = row["average_hours"] * row["avg_traffic"]
                attrs['distance_weight'] = row["distance"]
                attrs['weather_weight'] = row["avg_weather_risk"]
            
            # Add edge with all attributes
            G.add_edge(origin, destination, **attrs)
        
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        # Store this graph for future use
        self.graphs[optimize_for] = G
        
        return G
    
    def find_optimal_route(self, region_id, start_city, end_city, optimize_for="combined"):
        """
        Find optimal route visiting all cities in a region exactly once (TSP solution).
        
        Parameters:
        region_id (int): Region ID (1-5)
        start_city (str): Starting city ID
        end_city (str): Ending city ID
        optimize_for (str): Optimization objective ('time', 'distance', 'weather', 'combined')
        
        Returns:
        dict: Path and metrics
        """
        # For weather optimization, use the dedicated method that prioritizes lower risk
        if optimize_for == "weather":
            return self.find_weather_optimized_route(region_id, start_city, end_city)
            
        # For time optimization, use the dedicated method that prioritizes minimum time
        if optimize_for == "time":
            return self.find_time_optimized_route(region_id, start_city, end_city)
        
        print(f"\nFinding optimal route from {start_city} to {end_city} in region {region_id}...")
        print(f"Optimization strategy: {optimize_for}")
        
        # Check if region has changed
        if self.last_region != region_id:
            # Clear all graphs when region changes
            self.graphs = {
                "time": None,
                "distance": None,
                "weather": None,
                "combined": None
            }
            self.last_region = region_id
        
        # Always rebuild the graph for the requested optimization
        # to ensure we have the latest parameters
        self.build_graph(region_id, optimize_for)
        
        # Use the appropriate graph for this optimization
        self.G = self.graphs[optimize_for]
        
        # Get all cities in the region
        cities_in_region = [f"C{n}" for n in range((region_id - 1) * 10 + 1, region_id * 10 + 1)]
        
        # Check if all cities are in the graph
        missing_cities = [city for city in cities_in_region if city not in self.G.nodes]
        if missing_cities:
            print(f"Warning: Cities {missing_cities} are not in the graph.")
            # Remove missing cities from consideration
            cities_in_region = [city for city in cities_in_region if city in self.G.nodes]
        
        # Create a complete graph for the TSP problem
        # Where the weight between any two cities is the shortest path in the original graph
        tsp_graph = nx.Graph()
        
        # Add all cities as nodes
        for city in cities_in_region:
            tsp_graph.add_node(city)
        
        # Calculate shortest paths between all pairs of cities
        for i, city1 in enumerate(cities_in_region):
            for city2 in cities_in_region[i+1:]:
                if city1 != city2:
                    try:
                        # Find shortest path and its length
                        path_length = nx.shortest_path_length(
                            self.G, source=city1, target=city2, weight='weight'
                        )
                        path = nx.shortest_path(
                            self.G, source=city1, target=city2, weight='weight'
                        )
                        
                        # Add edge with weight and store the path
                        tsp_graph.add_edge(
                            city1, city2, 
                            weight=path_length,
                            path=path
                        )
                    except nx.NetworkXNoPath:
                        # If no path exists, don't add this edge
                        pass
        
        # Verify that all cities are connected in the TSP graph
        if not nx.is_connected(tsp_graph):
            print("Warning: Not all cities are connected in the region.")
            # Get the connected component containing the start city
            for component in nx.connected_components(tsp_graph):
                if start_city in component:
                    # Keep only cities in this component
                    cities_to_remove = set(cities_in_region) - component
                    for city in cities_to_remove:
                        if city in tsp_graph:
                            tsp_graph.remove_node(city)
                    cities_in_region = list(component)
                    break
        
        # Ensure both start and end cities are in the connected component
        if start_city not in tsp_graph or end_city not in tsp_graph:
            print(f"Error: Start or end city not connected to the network.")
            if start_city not in tsp_graph:
                print(f"Start city {start_city} is not connected.")
            if end_city not in tsp_graph:
                print(f"End city {end_city} is not connected.")
            # Return partial results
            return {
                "path": [start_city],
                "total_distance": 0,
                "total_time": 0,
                "avg_weather_risk": 0,
                "visited_cities": 1,
                "expected_cities": len(cities_in_region),
                "optimization": optimize_for,
                "error": "Start or end city not connected"
            }
        
        # Use a specialized TSP approach with fixed start/end
        current_city = start_city
        final_tsp_path = [start_city]
        unvisited = set(cities_in_region) - {start_city}
        
        # If end_city is specified and different from start_city,
        # we need to handle it specially
        if end_city != start_city:
            # Remove end_city from unvisited as we'll add it at the end
            if end_city in unvisited:
                unvisited.remove(end_city)
        
        # Visit all intermediate cities using greedy nearest neighbor
        while unvisited:
            next_city = None
            best_distance = float('inf')
            
            for candidate in unvisited:
                if tsp_graph.has_edge(current_city, candidate):
                    distance = tsp_graph[current_city][candidate]['weight']
                    if distance < best_distance:
                        best_distance = distance
                        next_city = candidate
            
            if next_city is None:
                print(f"Warning: No path found from {current_city} to any unvisited city.")
                break
            
            final_tsp_path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Add end_city to the path if it's not already the last city
        if end_city != start_city and end_city not in final_tsp_path:
            if tsp_graph.has_edge(current_city, end_city):
                final_tsp_path.append(end_city)
            else:
                print(f"Warning: No path found from {current_city} to end city {end_city}.")
        
        # Now convert the TSP path to the actual path through the original graph
        final_path = [final_tsp_path[0]]  # Start with the first city
        
        # For each consecutive pair of cities in the TSP path,
        # add the actual path between them
        for i in range(len(final_tsp_path) - 1):
            city1 = final_tsp_path[i]
            city2 = final_tsp_path[i + 1]
            
            # Get the actual path between these cities and skip the first city
            # to avoid duplication
            if tsp_graph.has_edge(city1, city2):
                actual_path = tsp_graph[city1][city2]['path'][1:]
                final_path.extend(actual_path)
            else:
                print(f"Warning: No path data between {city1} and {city2}.")
                final_path.append(city2)  # Just add the city if no path data
        
        # Extract unique cities from the path (for reporting)
        visited_cities = set(final_tsp_path)  # Use TSP path to count unique cities
        
        # Calculate metrics for the route
        total_distance = 0
        total_time = 0
        total_weather_risk = 0
        path_edges = []
        
        for i in range(len(final_path) - 1):
            source = final_path[i]
            target = final_path[i + 1]
            
            try:
                edge_data = self.G.get_edge_data(source, target)
                # Use the actual values for reporting, not the weighted ones
                total_distance += edge_data.get('distance', 0)
                total_time += edge_data.get('raw_time', 0) * edge_data.get('traffic', 1.0)
                total_weather_risk += edge_data.get('weather_risk', 3)
                path_edges.append(edge_data.get('route_id', f"{source}-{target}"))
            except:
                print(f"Warning: No direct edge between {source} and {target}.")
        
        # Calculate average weather risk
        avg_weather_risk = total_weather_risk / (len(final_path) - 1) if len(final_path) > 1 else 0
        
        # For result reporting, use the TSP path (which has each city exactly once)
        results = {
            "path": final_tsp_path,  # Report the simplified TSP path (each city once)
            "detailed_path": final_path,  # Store the detailed path if needed
            "route_ids": path_edges,
            "total_distance": round(total_distance, 2),
            "total_time": round(total_time, 2),
            "avg_weather_risk": round(avg_weather_risk, 2),
            "visited_cities": len(visited_cities),
            "expected_cities": len(cities_in_region),
            "optimization": optimize_for
        }
        
        print(f"Route planning complete.")
        print(f"Total distance: {results['total_distance']} miles")
        print(f"Total time: {results['total_time']} hours")
        print(f"Average weather risk: {results['avg_weather_risk']} (1-5 scale)")
        print(f"Visited {results['visited_cities']} out of {results['expected_cities']} cities")
        
        return results
    
    def compare_optimization_strategies(self, region_id, start_city, end_city):
        """Compare different optimization strategies"""
        
        strategies = ['time', 'distance', 'weather', 'combined']
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting optimization strategy: {strategy}")
            result = self.find_optimal_route(region_id, start_city, end_city, optimize_for=strategy)
            results[strategy] = result
        
        # Compile comparison
        comparison = pd.DataFrame({
            'Strategy': strategies,
            'Total Distance (miles)': [results[s]['total_distance'] for s in strategies],
            'Total Time (hours)': [results[s]['total_time'] for s in strategies],
            'Weather Risk (1-5)': [results[s]['avg_weather_risk'] for s in strategies],
            'Cities Visited': [results[s]['visited_cities'] for s in strategies],
        })
        
        print("\nStrategy Comparison:")
        print(comparison)
        
        return comparison, results
    
    def _calculate_road_quality(self, route_id, distance):
        """
        Calculate a deterministic road quality factor based on route ID and distance.
        This helps differentiate time vs distance routes without randomization.
        
        Values range from 0.7 (poor quality) to 1.0 (excellent quality)
        """
        # Use hash of route_id modulo 100 to get a number between 0-99
        route_hash = abs(hash(str(route_id))) % 100
        
        # Long distance routes tend to use highways (better quality)
        # Short distance routes might use local roads (poorer quality)
        distance_factor = min(1.0, distance / 250)  # Normalize with max at 250 miles
        
        # Combine factors (70% from route hash, 30% from distance)
        quality = 0.7 + (0.3 * route_hash / 100) + (0.3 * distance_factor)
        
        # Ensure quality is between 0.7 and 1.0
        return min(1.0, max(0.7, quality))
    
    def _get_region(self, city_id):
        """Extract region from city ID"""
        if not isinstance(city_id, str) or not city_id.startswith('C'):
            return "Unknown"
            
        try:
            city_num = int(city_id[1:])
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
            return "Unknown"
        except:
            return "Unknown"
    
    def _assign_weather_risk(self, description):
        """Assign risk level based on weather description"""
        if description in ["Clear", "Partly Cloudy", "Sunny"]:
            return 1
        elif description in ["Light Rain", "Overcast", "Mild Snow", "Cloudy"]:
            return 2
        elif description in ["Moderate Rain", "Moderate Snow", "Windy"]:
            return 3
        elif description in ["Heavy Rain", "Heavy Snow", "Sleet", "Freezing Rain"]:
            return 4
        elif description in ["Storm", "Fog", "Torrential Rain", "Thunderstorm", "Blizzard", "Hurricane", "Tornado"]:
            return 5
        return 3  # Default for unknown descriptions
    
    def _classify_traffic(self, no_of_vehicles, accident):
        """Classify traffic severity"""
        if accident:
            return 2.0
        elif no_of_vehicles >= 150:
            return 1.8
        elif no_of_vehicles >= 100:
            return 1.5
        elif no_of_vehicles >= 50:
            return 1.25
        return 1.0
    
    def _get_season(self, month):
        """Get season from month number"""
        if 3 <= month <= 5:
            return "spring"
        elif 6 <= month <= 8:
            return "summer"
        elif 9 <= month <= 11:
            return "fall"
        else:
            return "winter"