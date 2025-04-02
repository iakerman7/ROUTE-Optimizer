from flask import Flask, render_template, request, g
from ml_core import run_pipeline
import os

# ✅ Create Flask app first!
app = Flask(__name__)

# ✅ THEN attach before_first_request
@app.before_first_request
def setup_pipeline_cache():
    g.pipeline_cache = {}

# 📍 Coordinates for cities
city_coords = {
    # --- West (C1–C10) ---
    "C1": [37.7749, -122.4194],  "C2": [34.0522, -118.2437],  "C3": [36.1699, -115.1398],
    "C4": [47.6062, -122.3321],  "C5": [45.5152, -122.6784],  "C6": [39.7392, -104.9903],
    "C7": [32.7157, -117.1611],  "C8": [40.7608, -111.8910],  "C9": [35.6869, -105.9378],
    "C10": [38.5816, -121.4944],

    # --- Midwest (C11–C20) ---
    "C11": [41.8781, -87.6298],  "C12": [39.0997, -94.5786],  "C13": [43.0389, -87.9065],
    "C14": [41.2565, -95.9345],  "C15": [42.3314, -83.0458],  "C16": [44.9537, -93.0900],
    "C17": [39.7684, -86.1581],  "C18": [39.9612, -82.9988],  "C19": [41.6611, -91.5302],
    "C20": [40.6936, -89.5889],

    # --- South (C21–C30) ---
    "C21": [29.7604, -95.3698],  "C22": [32.7767, -96.7970],  "C23": [30.2672, -97.7431],
    "C24": [29.4241, -98.4936],  "C25": [35.4676, -97.5164],  "C26": [36.1539, -95.9928],
    "C27": [33.7488, -84.3880],  "C28": [32.7555, -97.3308],  "C29": [34.7465, -92.2896],
    "C30": [30.3322, -81.6557],

    # --- Northeast (C31–C40) ---
    "C31": [42.3601, -71.0589],  "C32": [40.7128, -74.0060],  "C33": [39.9526, -75.1652],
    "C34": [38.9072, -77.0369],  "C35": [41.8240, -71.4128],  "C36": [43.6615, -70.2553],
    "C37": [42.0987, -75.9179],  "C38": [40.2732, -76.8867],  "C39": [40.4406, -79.9959],
    "C40": [41.2033, -77.1945],

    # --- Southeast (C41–C50) ---
    "C41": [35.2271, -80.8431],  "C42": [33.7490, -84.3880],  "C43": [32.7765, -79.9311],
    "C44": [36.8508, -76.2859],  "C45": [35.0456, -85.3097],  "C46": [36.1627, -86.7816],
    "C47": [30.6954, -88.0399],  "C48": [30.3322, -81.6557],  "C49": [33.5207, -86.8025],
    "C50": [27.9506, -82.4572],
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        region_id = int(request.form["region"])
        start_city = request.form["start_city"].upper()
        destination_city = request.form["destination_city"].upper()
        optimize_for = request.form["optimize_for"]

        key = f"{region_id}-{start_city}-{destination_city}-{optimize_for}"
        if not hasattr(g, 'pipeline_cache'):
            g.pipeline_cache = {}
        if key not in g.pipeline_cache:
            g.pipeline_cache[key] = run_pipeline(region_id, start_city, destination_city, optimize_for)

        result = g.pipeline_cache[key]
        result["optimized_for"] = optimize_for
        result["region_name"] = {
            1: "West", 2: "Midwest", 3: "South", 4: "Northeast", 5: "Southeast"
        }[region_id]

        return render_template("index.html", result=result, city_coords=city_coords)

    return render_template("index.html", result=None, city_coords=city_coords)

# ✅ Correct app run block for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
