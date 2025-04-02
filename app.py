from flask import Flask, render_template, request
from ml_core import run_pipeline

app = Flask(__name__)

# Dummy lat/lon for cities (you'll replace with real ones or load from CSV)
city_coords = {
    # --- West (C1–C10) ---
   "C1": [37.7749, -122.4194],  # San Francisco, CA
   "C2": [34.0522, -118.2437],  # Los Angeles, CA
   "C3": [36.1699, -115.1398],  # Las Vegas, NV
   "C4": [47.6062, -122.3321],  # Seattle, WA
   "C5": [45.5152, -122.6784],  # Portland, OR
   "C6": [39.7392, -104.9903],  # Denver, CO
   "C7": [32.7157, -117.1611],  # San Diego, CA
   "C8": [40.7608, -111.8910],  # Salt Lake City, UT
   "C9": [35.6869, -105.9378],  # Santa Fe, NM
   "C10": [38.5816, -121.4944], # Sacramento, CA

   # --- Midwest (C11–C20) ---
   "C11": [41.8781, -87.6298],  # Chicago, IL
   "C12": [39.0997, -94.5786],  # Kansas City, MO
   "C13": [43.0389, -87.9065],  # Milwaukee, WI
   "C14": [41.2565, -95.9345],  # Omaha, NE
   "C15": [42.3314, -83.0458],  # Detroit, MI
   "C16": [44.9537, -93.0900],  # St. Paul, MN
   "C17": [39.7684, -86.1581],  # Indianapolis, IN
   "C18": [39.9612, -82.9988],  # Columbus, OH
   "C19": [41.6611, -91.5302],  # Iowa City, IA
   "C20": [40.6936, -89.5889],  # Peoria, IL

   # --- South (C21–C30) ---
   "C21": [29.7604, -95.3698],  # Houston, TX
   "C22": [32.7767, -96.7970],  # Dallas, TX
   "C23": [30.2672, -97.7431],  # Austin, TX
   "C24": [29.4241, -98.4936],  # San Antonio, TX
   "C25": [35.4676, -97.5164],  # Oklahoma City, OK
   "C26": [36.1539, -95.9928],  # Tulsa, OK
   "C27": [33.7488, -84.3880],  # Atlanta, GA
   "C28": [32.7555, -97.3308],  # Fort Worth, TX
   "C29": [34.7465, -92.2896],  # Little Rock, AR
   "C30": [30.3322, -81.6557],  # Jacksonville, FL

   # --- Northeast (C31–C40) ---
   "C31": [42.3601, -71.0589],  # Boston, MA
   "C32": [40.7128, -74.0060],  # New York, NY
   "C33": [39.9526, -75.1652],  # Philadelphia, PA
   "C34": [38.9072, -77.0369],  # Washington, D.C.
   "C35": [41.8240, -71.4128],  # Providence, RI
   "C36": [43.6615, -70.2553],  # Portland, ME
   "C37": [42.0987, -75.9179],  # Binghamton, NY
   "C38": [40.2732, -76.8867],  # Harrisburg, PA
   "C39": [40.4406, -79.9959],  # Pittsburgh, PA
   "C40": [41.2033, -77.1945],  # Williamsport, PA

   # --- Southeast (C41–C50) ---
   "C41": [35.2271, -80.8431],  # Charlotte, NC
   "C42": [33.7490, -84.3880],  # Atlanta, GA
   "C43": [32.7765, -79.9311],  # Charleston, SC
   "C44": [36.8508, -76.2859],  # Norfolk, VA
   "C45": [35.0456, -85.3097],  # Chattanooga, TN
   "C46": [36.1627, -86.7816],  # Nashville, TN
   "C47": [30.6954, -88.0399],  # Mobile, AL
   "C48": [30.3322, -81.6557],  # Jacksonville, FL
   "C49": [33.5207, -86.8025],  # Birmingham, AL
   "C50": [27.9506, -82.4572],  # Tampa, FL
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        region_id = int(request.form["region"])
        start_city = request.form["start_city"].upper()
        destination_city = request.form["destination_city"].upper()
        optimize_for = request.form["optimize_for"]

        result = run_pipeline(region_id, start_city, destination_city, optimize_for)
        result["optimized_for"] = optimize_for
        result["region_name"] = {
            1: "West", 2: "Midwest", 3: "South", 4: "Northeast", 5: "Southeast"
        }[region_id]

        return render_template("index.html", result=result, city_coords=city_coords)

    return render_template("index.html", result=None, city_coords=city_coords)

if __name__ == "__main__":
    app.run(debug=True)

