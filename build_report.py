"""Build the SpaceX Falcon 9 Data Science Capstone report as a 16:9 PDF."""
import os, sqlite3, textwrap, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

D = os.environ.get("DATA", os.environ.get("TMPDIR", "."))
GITHUB = os.environ.get("GITHUB_URL", "https://github.com/TrungKiencding/IBM-Data-Science-Capstone-SpaceX")
NAVY, BLUE, ORANGE, GREEN, RED, GREY = "#0B1F3A", "#1F6FEB", "#F28C28", "#2DA44E", "#D1242F", "#57606A"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(f"{D}/dataset_part_2.csv")
X_raw = pd.read_csv(f"{D}/dataset_part_3.csv")
geo = pd.read_csv(f"{D}/spacex_launch_geo.csv")
dash = pd.read_csv(f"{D}/spacex_launch_dash.csv")
df["Year"] = pd.to_datetime(df["Date"]).dt.year

pdf = PdfPages("SpaceX_Capstone_Report.pdf")
page = [0]

def slide(title, section=None, dark=False):
    fig = plt.figure(figsize=(13.33, 7.5))
    bg = NAVY if dark else "white"
    fig.patch.set_facecolor(bg)
    if not dark:
        fig.add_artist(plt.Rectangle((0, 0.9), 1, 0.1, color=NAVY, transform=fig.transFigure))
        fig.text(0.04, 0.945, title, fontsize=24, color="white", weight="bold", va="center")
        if section:
            fig.text(0.96, 0.945, section.upper(), fontsize=10, color=ORANGE, ha="right", va="center", weight="bold")
    page[0] += 1
    fig.text(0.96, 0.02, str(page[0]), fontsize=9, color=GREY, ha="right")
    return fig

def bullets(fig, items, x=0.05, y=0.83, w=0.9, size=15, gap=0.012, wrap=None):
    wrap = wrap or int(115 * w / 0.9 * 15 / size)
    for it in items:
        sub = it.startswith("-")
        txt = it.lstrip("- ")
        lines = textwrap.wrap(txt, wrap - (6 if sub else 0))
        fig.text(x + (0.03 if sub else 0), y, ("–  " if sub else "•  ") + lines[0], fontsize=size - (2 if sub else 0),
                 color=GREY if sub else "#1F2328", va="top")
        for ln in lines[1:]:
            y -= size * 0.0028
            fig.text(x + (0.03 if sub else 0) + 0.018, y, ln, fontsize=size - (2 if sub else 0), color=GREY if sub else "#1F2328", va="top")
        y -= size * 0.0028 + gap
    return y

def save(fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)

def section(title, sub):
    fig = slide(title, dark=True)
    fig.text(0.07, 0.55, title, fontsize=40, color="white", weight="bold")
    fig.text(0.07, 0.46, sub, fontsize=18, color=ORANGE)
    save(fig)

def insight(fig, text, y=0.07):
    fig.text(0.05, y, "Insight: " + text, fontsize=13, color=NAVY, weight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF4E5", ec=ORANGE))

def table(fig, rect, data, cols, size=11, colw=None):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=data, colLabels=cols, loc="upper center", cellLoc="left", colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(size); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if r == 0: cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif r % 2 == 0: cell.set_facecolor("#F6F8FA")
    return ax

def img(fig, path, rect):
    ax = fig.add_axes(rect); ax.imshow(plt.imread(path)); ax.axis("off")
    for sp in ax.spines.values(): sp.set_visible(True)

def code(fig, rect, sql):
    fig.text(rect[0], rect[1], sql, fontsize=11, family="monospace", color="white", va="top",
             bbox=dict(boxstyle="round,pad=0.8", fc="#24292F", ec="none"))

# ---------------------------------------------------------------- ML
Y = df["Class"].to_numpy()
X = StandardScaler().fit_transform(X_raw)
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=2)
models = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.01, 0.1, 1], "penalty": ["l2"], "solver": ["lbfgs"]}),
    "SVM": (SVC(), {"kernel": ["linear", "rbf", "poly", "sigmoid"], "C": np.logspace(-3, 3, 5), "gamma": np.logspace(-3, 3, 5)}),
    "Decision Tree": (DecisionTreeClassifier(random_state=2), {"criterion": ["gini", "entropy"], "splitter": ["best", "random"],
                      "max_depth": [2 * n for n in range(1, 10)], "max_features": ["sqrt"], "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 5, 10]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": list(range(1, 11)), "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p": [1, 2]}),
}
res = {}
for name, (m, grid) in models.items():
    g = GridSearchCV(m, grid, cv=10).fit(X_tr, y_tr)
    res[name] = dict(cv=g.best_score_, test=g.score(X_te, y_te), params=g.best_params_, cm=confusion_matrix(y_te, g.predict(X_te)))
best = max(res, key=lambda k: (res[k]["test"], res[k]["cv"]))

# ---------------------------------------------------------------- SQL
con = sqlite3.connect(":memory:")
pd.read_csv(f"{D}/Spacex.csv").to_sql("SPACEXTBL", con, index=False)
q = lambda s: pd.read_sql(s, con)

# ================================================================ SLIDES
fig = slide("", dark=True)
fig.text(0.07, 0.62, "Winning Space Race with Data Science", fontsize=38, color="white", weight="bold")
fig.text(0.07, 0.53, "Predicting SpaceX Falcon 9 first-stage landing success", fontsize=20, color=ORANGE)
fig.text(0.07, 0.36, "IBM Data Science Professional Certificate — Applied Data Science Capstone", fontsize=14, color="#C9D1D9")
fig.text(0.07, 0.31, f"GitHub: {GITHUB}", fontsize=12, color="#C9D1D9")
fig.text(0.07, 0.26, "September 2026", fontsize=12, color="#C9D1D9")
save(fig)

fig = slide("Outline")
bullets(fig, ["Executive Summary", "Introduction", "Methodology", "Results",
              "- Exploratory data analysis (visualization & SQL)", "- Interactive analytics (Folium map, Plotly Dash)",
              "- Predictive analysis (classification)", "Conclusion", "Appendix"], size=18, gap=0.02)
save(fig)

fig = slide("Executive Summary")
bullets(fig, [
    "Methodologies",
    "- Data collection via the SpaceX REST API and web scraping of Wikipedia's Falcon 9 launch tables",
    "- Data wrangling: missing-value handling, one-hot encoding, creation of the binary landing label 'Class'",
    "- EDA with pandas/Matplotlib/Seaborn and SQL; interactive analytics with Folium and Plotly Dash",
    "- Predictive modelling: Logistic Regression, SVM, Decision Tree and KNN tuned with 10-fold GridSearchCV",
    "Results",
    f"- Overall landing success rate is {df['Class'].mean():.0%}; it rose from 0% (2010–2012) to over 80% after 2017",
    "- KSC LC-39A has the most successful launches; ES-L1, GEO, HEO and SSO orbits reached 100% success",
    f"- All four models reach ~{res[best]['test']:.0%} test accuracy; {best} was selected "
    f"(CV accuracy {res[best]['cv']:.1%}) — main error mode is false positives",
], size=15, gap=0.018)
save(fig)

fig = slide("Introduction")
bullets(fig, [
    "Project background and context",
    "- SpaceX advertises Falcon 9 launches at ~$62 million, while competitors charge upward of $165 million.",
    "- Most of the saving comes from re-using the first stage. If we can predict whether the first stage will land, "
    "we can estimate the cost of a launch — valuable for a competitor ('SpaceY') bidding against SpaceX.",
    "Problems we want to answer",
    "- Which factors (payload, orbit, launch site, booster version, flight number…) influence landing success?",
    "- How do these factors interact, and how has success evolved over time?",
    "- What operating conditions are needed for a successful landing program?",
    "- Can a machine-learning model reliably predict landing success for a new launch?",
], size=16, gap=0.02)
save(fig)

section("Section 1", "Methodology")

fig = slide("Methodology", "Methodology")
bullets(fig, [
    "Data collection: SpaceX REST API (api.spacexdata.com/v4) + web scraping of Wikipedia with BeautifulSoup",
    "Data wrangling: filtered to Falcon 9, replaced missing PayloadMass with the mean, derived 'Class' label "
    "from 'Outcome' (True ASDS / RTLS / Ocean → 1, others → 0)",
    "Exploratory data analysis (EDA) using visualization (scatter, bar, line plots) and SQL (SQLite / Db2)",
    "Interactive visual analytics using Folium (launch-site maps, marker clusters, distance lines) and Plotly Dash",
    "Predictive analysis using classification models",
    "- Standardize features → train/test split (80/20) → GridSearchCV (cv=10) → compare accuracy & confusion matrices",
], size=16, gap=0.025)
save(fig)

def flow(fig, steps, y=0.55, color=BLUE):
    n = len(steps); w = 0.9 / n
    for i, s in enumerate(steps):
        x = 0.05 + i * w
        fig.add_artist(matplotlib.patches.FancyBboxPatch((x + 0.005, y - 0.09), w - 0.03, 0.18, boxstyle="round,pad=0.01",
                        fc=color, ec="none", transform=fig.transFigure))
        fig.text(x + w / 2 - 0.01, y, "\n".join(textwrap.wrap(s, 18)), ha="center", va="center", color="white", fontsize=12, weight="bold")
        if i < n - 1:
            fig.text(x + w - 0.018, y, "➜", fontsize=18, color=GREY, ha="center", va="center")

fig = slide("Data Collection – SpaceX API", "Methodology")
flow(fig, ["GET /v4/launches/past", "Normalize JSON (json_normalize)", "Enrich IDs: rocket, launchpad, payload, core",
           "Build DataFrame & filter Falcon 9", "Replace missing PayloadMass with mean", "Export dataset_part_1.csv"], y=0.62)
bullets(fig, ["Helper functions getBoosterVersion(), getLaunchSite(), getPayloadData(), getCoreData() call extra endpoints "
              "to resolve IDs into readable values.",
              "Result: 90 Falcon 9 launches with 17 columns (FlightNumber, Date, BoosterVersion, PayloadMass, Orbit, LaunchSite, "
              "Outcome, Flights, GridFins, Reused, Legs, LandingPad, Block, ReusedCount, Serial, Longitude, Latitude).",
              f"Notebook: {GITHUB}/blob/main/1_jupyter-labs-spacex-data-collection-api.ipynb"], y=0.38, size=13)
save(fig)

fig = slide("Data Collection – Web Scraping", "Methodology")
flow(fig, ["Request Wikipedia 'List of Falcon 9 and Falcon Heavy launches'", "Parse HTML with BeautifulSoup",
           "Extract column names from <th>", "Parse launch tables row by row", "Build dict → pandas DataFrame", "Export spacex_web_scraped.csv"],
     y=0.62, color=ORANGE)
bullets(fig, ["Captured Flight No., Date, Time, Version Booster, Launch Site, Payload, Payload mass, Orbit, Customer, "
              "Launch outcome and Booster landing for 121 launches.",
              f"Notebook: {GITHUB}/blob/main/2_jupyter-labs-webscraping.ipynb"], y=0.38, size=13)
save(fig)

fig = slide("Data Wrangling", "Methodology")
flow(fig, ["Load dataset_part_1", "Check missing values & dtypes", "Count launches per site & per orbit",
           "Count landing outcomes", "Map outcomes → Class (0/1)", "Export dataset_part_2.csv"], y=0.62, color=GREEN)
bullets(fig, ["Landing outcomes: True ASDS, True RTLS, True Ocean = success (1); False ASDS, False RTLS, False Ocean, "
              "None ASDS, None None = failure (0).",
              f"Resulting average success rate: {df['Class'].mean():.1%}.",
              f"Notebook: {GITHUB}/blob/main/3_labs-jupyter-spacex-Data-wrangling.ipynb"], y=0.38, size=13)
save(fig)

fig = slide("EDA with Data Visualization", "Methodology")
bullets(fig, [
    "Scatter plots — relationship between two variables coloured by landing outcome:",
    "- Flight Number vs Launch Site, Payload vs Launch Site, Flight Number vs Orbit, Payload vs Orbit",
    "Bar chart — success rate of each orbit type (compare categorical groups)",
    "Line chart — yearly average success rate (show trend over time)",
    "Feature engineering: one-hot encoding of Orbit, LaunchSite, LandingPad, Serial → 83 numeric features (dataset_part_3.csv)",
    f"Notebook: {GITHUB}/blob/main/5_edadataviz.ipynb",
], size=15, gap=0.03)
save(fig)

fig = slide("EDA with SQL", "Methodology")
bullets(fig, [
    "Loaded Spacex.csv into a SQL database (SQLite, table SPACEXTBL) and queried with %sql magic:",
    "- Unique launch sites; 5 records whose site begins with 'CCA'",
    "- Total payload of NASA (CRS) missions; average payload of booster F9 v1.1",
    "- Date of first successful ground-pad landing",
    "- Boosters with drone-ship success and payload 4000–6000 kg",
    "- Count of successful / failed mission outcomes",
    "- Boosters that carried the maximum payload (sub-query)",
    "- 2015 drone-ship failures by month; ranking of landing outcomes 2010-06-04 → 2017-03-20",
    f"Notebook: {GITHUB}/blob/main/4_jupyter-labs-eda-sql-coursera_sqllite.ipynb",
], size=15, gap=0.02)
save(fig)

fig = slide("Build an Interactive Map with Folium", "Methodology")
bullets(fig, [
    "Circles + text markers for each launch site at its latitude/longitude (NASA JSC as reference point)",
    "MarkerCluster with green (success) / red (failure) markers for every launch record at each site",
    "MousePosition plugin to read coordinates, and PolyLines with distance labels (Haversine) from a launch site "
    "to the nearest coastline, railway, highway and city",
    "Why: to see whether launch sites are near the equator/coast and how proximity relates to safety & logistics",
    f"Notebook: {GITHUB}/blob/main/6_lab_jupyter_launch_site_location.ipynb",
], size=15, gap=0.03)
save(fig)

fig = slide("Build a Dashboard with Plotly Dash", "Methodology")
bullets(fig, [
    "Dropdown to select All Sites or an individual launch site",
    "Pie chart: total successful launches by site (All) or success vs failure for a chosen site",
    "RangeSlider to filter payload mass (0 – 10 000 kg)",
    "Scatter chart: payload mass vs outcome, coloured by booster version category",
    "Why: lets stakeholders interactively explore which sites, payload ranges and boosters give the highest success",
    f"Code: {GITHUB}/blob/main/spacex_dash_app.py",
], size=15, gap=0.03)
save(fig)

fig = slide("Predictive Analysis (Classification)", "Methodology")
flow(fig, ["Load features (83) & Class", "StandardScaler", "train_test_split 80/20, random_state=2",
           "GridSearchCV cv=10 for LR, SVM, Tree, KNN", "Evaluate test accuracy & confusion matrix", "Select best model"], y=0.62, color=NAVY)
bullets(fig, ["Hyper-parameters searched: LR (C), SVM (kernel, C, gamma), Decision Tree (criterion, splitter, depth, leaf/split sizes), "
              "KNN (n_neighbors, algorithm, p).",
              f"Notebook: {GITHUB}/blob/main/7_SpaceX_Machine_Learning_Prediction_Part_5.ipynb"], y=0.38, size=13)
save(fig)

fig = slide("Results", "Overview")
bullets(fig, [
    f"Exploratory data analysis: success rate improves strongly with flight number/experience; {df['Class'].mean():.0%} overall",
    "Interactive analytics: all sites are close to the coast and far from cities; KSC LC-39A is the most successful site",
    f"Predictive analysis: tuned classifiers reach {res[best]['test']:.1%} test accuracy; {best} chosen as best model",
], size=16, gap=0.04)
save(fig)

# ---------------------------------------------------------------- EDA VIS
section("Section 2", "Insights drawn from EDA")
cmap = {0: RED, 1: GREEN}

def scatter_slide(title, x, y, xlabel, ylabel, text):
    fig = slide(title, "EDA – Visualization")
    ax = fig.add_axes([0.12, 0.2, 0.83, 0.64])
    for c in (0, 1):
        s = df[df.Class == c]
        ax.scatter(s[x], s[y], c=cmap[c], s=45, alpha=0.8, label="Landed" if c else "Failed", edgecolor="white")
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=12); ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.08))
    ax.grid(axis="x", alpha=0.3)
    insight(fig, text); save(fig)

scatter_slide("Flight Number vs. Launch Site", "FlightNumber", "LaunchSite", "Flight Number", "Launch Site",
              "Early flights (mostly CCAFS SLC 40) failed; success grows with flight number at every site.")
scatter_slide("Payload vs. Launch Site", "PayloadMass", "LaunchSite", "Payload Mass (kg)", "Launch Site",
              "VAFB SLC 4E launched no payload > 10 000 kg; heavy payloads (>7 000 kg) mostly land successfully.")

fig = slide("Success Rate vs. Orbit Type", "EDA – Visualization")
ax = fig.add_axes([0.08, 0.2, 0.88, 0.64])
orb = df.groupby("Orbit")["Class"].mean().sort_values(ascending=False)
ax.bar(orb.index, orb.values, color=[GREEN if v == 1 else BLUE for v in orb.values])
for i, v in enumerate(orb.values): ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.12)
insight(fig, "ES-L1, GEO, HEO and SSO reach 100% success; GTO (~52%) and SO (0%) are the hardest orbits.")
save(fig)

scatter_slide("Flight Number vs. Orbit Type", "FlightNumber", "Orbit", "Flight Number", "Orbit",
              "In LEO success follows flight number; in GTO there is no clear relationship with flight number.")
scatter_slide("Payload vs. Orbit Type", "PayloadMass", "Orbit", "Payload Mass (kg)", "Orbit",
              "Heavy payloads have higher success for Polar, LEO and ISS; GTO mixes outcomes across payloads.")

fig = slide("Launch Success Yearly Trend", "EDA – Visualization")
ax = fig.add_axes([0.08, 0.2, 0.88, 0.64])
yr = df.groupby("Year")["Class"].mean()
ax.plot(yr.index, yr.values, marker="o", color=BLUE, lw=3)
for x_, v in yr.items(): ax.text(x_, v + 0.04, f"{v:.0%}", ha="center", fontsize=10)
ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.1); ax.set_xticks(yr.index); ax.grid(alpha=0.3)
insight(fig, "Success rate kept rising from 2013 to 2020 (small dip in 2018) — SpaceX learns quickly.")
save(fig)

# ---------------------------------------------------------------- SQL
expl = {
 "All Launch Site Names": "DISTINCT removes duplicates so each launch site appears once: CCAFS LC-40, CCAFS SLC-40, KSC LC-39A, VAFB SLC-4E.",
 "Total Payload Mass": "SUM aggregates PAYLOAD_MASS__KG_ over all rows whose customer is NASA (CRS) – the ISS resupply missions.",
 "Average Payload Mass by F9 v1.1": "AVG computes the mean payload of the F9 v1.1 booster, an early version with modest lift capacity.",
 "First Successful Ground Landing Date": "MIN(Date) on rows with 'Success (ground pad)' finds the first RTLS landing – a milestone for reusability.",
 "Successful Drone Ship Landing with Payload 4000–6000": "Combines the landing-outcome filter with a payload range using AND to list qualifying boosters.",
 "Total Number of Successful and Failure Mission Outcomes": "GROUP BY Mission_Outcome with COUNT(*) shows mission (not landing) reliability is ~99%.",
 "Boosters Carried Maximum Payload": "A sub-query returns MAX(payload); the outer query lists every booster that carried it (15,600 kg).",
 "2015 Launch Records": "substr() extracts month and year from the text date because SQLite has no month-name function.",
 "Rank Landing Outcomes Between 2010-06-04 and 2017-03-20": "GROUP BY + ORDER BY COUNT DESC ranks landing outcomes in the early period of the program.",
}
sql_items = [
    ("All Launch Site Names", "SELECT DISTINCT Launch_Site\nFROM SPACEXTBL;", "There are four unique launch sites in the data."),
    ("Launch Site Names Begin with 'CCA'",
     "SELECT * FROM SPACEXTBL\nWHERE Launch_Site LIKE 'CCA%'\nLIMIT 5;", "The first five CCA launches were all early LEO missions for SpaceX/NASA."),
    ("Total Payload Mass", "SELECT SUM(PAYLOAD_MASS__KG_) AS Total_Payload\nFROM SPACEXTBL\nWHERE Customer = 'NASA (CRS)';",
     "Total payload carried by boosters launched for NASA (CRS)."),
    ("Average Payload Mass by F9 v1.1", "SELECT AVG(PAYLOAD_MASS__KG_) AS Avg_Payload\nFROM SPACEXTBL\nWHERE Booster_Version = 'F9 v1.1';",
     "Average payload mass carried by booster version F9 v1.1."),
    ("First Successful Ground Landing Date", "SELECT MIN(Date) AS First_Success\nFROM SPACEXTBL\nWHERE Landing_Outcome = 'Success (ground pad)';",
     "The first successful ground-pad landing happened in December 2015."),
    ("Successful Drone Ship Landing with Payload 4000–6000",
     "SELECT Booster_Version FROM SPACEXTBL\nWHERE Landing_Outcome = 'Success (drone ship)'\n  AND PAYLOAD_MASS__KG_ > 4000\n  AND PAYLOAD_MASS__KG_ < 6000;",
     "Four F9 FT boosters landed on a drone ship with a 4000–6000 kg payload."),
    ("Total Number of Successful and Failure Mission Outcomes",
     "SELECT Mission_Outcome, COUNT(*) AS Total\nFROM SPACEXTBL\nGROUP BY Mission_Outcome;", "Almost every mission succeeded; only one in-flight failure."),
    ("Boosters Carried Maximum Payload",
     "SELECT DISTINCT Booster_Version FROM SPACEXTBL\nWHERE PAYLOAD_MASS__KG_ = (\n  SELECT MAX(PAYLOAD_MASS__KG_)\n  FROM SPACEXTBL);",
     "All boosters carrying the max payload are Block 5 (F9 B5) versions."),
    ("2015 Launch Records",
     "SELECT substr(Date,6,2) AS Month, Landing_Outcome,\n       Booster_Version, Launch_Site\nFROM SPACEXTBL\nWHERE substr(Date,1,4) = '2015'\n  AND Landing_Outcome = 'Failure (drone ship)';",
     "Two drone-ship landing failures in 2015 (January and April), both from CCAFS LC-40."),
    ("Rank Landing Outcomes Between 2010-06-04 and 2017-03-20",
     "SELECT Landing_Outcome, COUNT(*) AS Cnt\nFROM SPACEXTBL\nWHERE Date BETWEEN '2010-06-04' AND '2017-03-20'\nGROUP BY Landing_Outcome\nORDER BY Cnt DESC;",
     "'No attempt' dominates this early period, followed by drone-ship successes and failures."),
]
for title, sql, note in sql_items:
    fig = slide(title, "EDA – SQL")
    code(fig, (0.04, 0.84), sql)
    r = q(sql)
    if not title.startswith("Launch Site Names Begin"):
        fig.text(0.04, 0.42, "Explanation", fontsize=14, weight="bold", color=NAVY)
        bullets(fig, [expl[title], f"Rows returned: {len(r)}"], x=0.04, y=0.37, w=0.46, size=12)
    if title.startswith("Launch Site Names Begin"):
        r = r[["Date", "Booster_Version", "Launch_Site", "Payload", "PAYLOAD_MASS__KG_", "Orbit", "Customer", "Landing_Outcome"]]
        r["Payload"] = r["Payload"].str.slice(0, 28)
        table(fig, [0.04, 0.12, 0.92, 0.45], r.values.tolist(), list(r.columns), size=9)
    else:
        r = r.round(2)
        table(fig, [0.55, 0.15, 0.41, 0.7], r.astype(str).values.tolist(), list(r.columns), size=11)
    insight(fig, note, y=0.06)
    save(fig)

# ---------------------------------------------------------------- FOLIUM
section("Section 3", "Launch Sites Proximities Analysis")
sites = geo.groupby("Launch Site").agg(Lat=("Lat", "first"), Long=("Long", "first"), n=("class", "size"), s=("class", "sum"))

SH = os.environ.get("SHOTS", "shots")
for title, f, pts in [
  ("All Launch Sites on a Global Map", "map1", ["All 4 launch sites are in the USA: 3 in Florida (east coast) and VAFB SLC-4E in California (west coast).",
     "Sites are as far south as possible – closer to the equator, where Earth's rotation gives extra velocity and saves fuel.",
     "All sites are on the coast so failed rockets fall into the ocean, away from people."]),
  ("Launch Outcomes per Site (Colour-labelled Markers)", "map2", ["Each launch is a marker in a MarkerCluster: green = successful landing, red = failure.",
     "KSC LC-39A (left) shows mostly green markers – highest success ratio (76.9%).",
     "CCAFS LC-40 (right) has the most launches but many red markers, mainly early flights."]),
  ("Distances from CCAFS SLC-40 to Proximities", "map3", ["Distances computed with the Haversine formula and drawn as PolyLines:",
     "- Coastline 0.86 km, Highway 0.56 km, Railway 1.28 km → easy transport of heavy parts and safe ocean trajectory.",
     "- Nearest city (Melbourne) 51.4 km → far from populated areas for safety."])]:
    fig = slide(title, "Folium")
    img(fig, f"{SH}/{f}.png", [0.03, 0.08, 0.58, 0.8])
    bullets(fig, pts, x=0.63, y=0.82, w=0.30, size=12, gap=0.03)
    save(fig)

# ---------------------------------------------------------------- DASH
section("Section 4", "Build a Dashboard with Plotly Dash")
for title, f, pts in [
  ("Launch Success Count for All Sites", "dash1", ["Dropdown = 'All Sites': pie of successful launches by site.",
     "KSC LC-39A has the largest share of successes (41.7%), followed by CCAFS LC-40 (29.2%).",
     "CCAFS SLC-40 has the smallest share (12.5%)."]),
  ("Site with Highest Launch Success Ratio – KSC LC-39A", "dash2", ["Dropdown = 'KSC LC-39A': success (1) vs failure (0).",
     "76.9% success (10 of 13 launches) – the best ratio of all sites.",
     "This site hosts most of the newer FT/B4/B5 boosters."]),
  ("Payload vs. Launch Outcome – All Sites", "dash3", ["RangeSlider 0–10,000 kg, points coloured by booster version.",
     "Payloads 2,000–5,500 kg have the highest success rate; > 5,500 kg few successes.",
     "FT booster has the most successes; v1.0 and v1.1 mostly failed."])]:
    fig = slide(title, "Dash")
    img(fig, f"{SH}/{f}.png", [0.03, 0.08, 0.6, 0.8])
    bullets(fig, pts, x=0.65, y=0.82, w=0.28, size=12, gap=0.03)
    save(fig)

# ---------------------------------------------------------------- ML
section("Section 5", "Predictive Analysis (Classification)")
fig = slide("Classification Accuracy", "Classification")
ax = fig.add_axes([0.08, 0.2, 0.55, 0.62])
names = list(res); xi = np.arange(len(names))
ax.bar(xi - 0.2, [res[n]["cv"] for n in names], 0.4, label="CV accuracy (train)", color=BLUE)
ax.bar(xi + 0.2, [res[n]["test"] for n in names], 0.4, label="Test accuracy", color=ORANGE)
for i, n in enumerate(names):
    ax.text(i - 0.2, res[n]["cv"] + 0.01, f"{res[n]['cv']:.3f}", ha="center", fontsize=10)
    ax.text(i + 0.2, res[n]["test"] + 0.01, f"{res[n]['test']:.3f}", ha="center", fontsize=10)
ax.set_xticks(xi); ax.set_xticklabels(names); ax.set_ylim(0.5, 1.02); ax.legend(frameon=False, loc="upper left")
table(fig, [0.66, 0.3, 0.31, 0.5], [[n, "\n".join(textwrap.wrap(", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
       for k, v in res[n]["params"].items()), 30))] for n in names], ["Model", "Best parameters"], size=8, colw=[0.3, 0.7])
insight(fig, f"All models score {res[best]['test']:.3f} on the small test set (18 samples); {best} has the best CV accuracy "
             f"({res[best]['cv']:.3f}).")
save(fig)

fig = slide(f"Confusion Matrix – {best}", "Classification")
ax = fig.add_axes([0.08, 0.15, 0.4, 0.68])
cm = res[best]["cm"]; ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=26, color="white" if v > cm.max() / 2 else NAVY)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["did not land", "landed"]); ax.set_yticklabels(["did not land", "landed"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
tn, fp, fn, tp = cm.ravel()
bullets(fig, [f"True positives: {tp}, true negatives: {tn}", f"False positives: {fp}, false negatives: {fn}",
              "The model distinguishes the classes well; the main problem is false positives — some failed landings are "
              "predicted as successful.", "Precision (landed): " + f"{tp / (tp + fp):.2f}, Recall: {tp / (tp + fn):.2f}"],
        x=0.55, y=0.75, w=0.4, size=14, gap=0.03)
save(fig)

# ---------------------------------------------------------------- CONCLUSION
fig = slide("Conclusions")
bullets(fig, [
    "The larger the flight number (experience), the higher the landing success rate at every launch site.",
    "Success rate rose from 0% in 2010–2013 to ~80%+ from 2017 onward.",
    "Orbits ES-L1, GEO, HEO and SSO have 100% success; GTO is the most difficult.",
    "KSC LC-39A has the most successful launches and the highest success ratio among sites.",
    "Launches with payloads below ~5 000 kg and FT / B4 / B5 boosters are more likely to land.",
    "Launch sites sit near the equator line and the coast, and far from cities.",
    f"{best} is the chosen predictor ({res[best]['test']:.1%} test accuracy) — "
    "SpaceY can use it to estimate first-stage recovery and thus SpaceX's launch cost when bidding.",
], size=16, gap=0.025)
save(fig)

fig = slide("Innovative Insights")
bullets(fig, [
    "Learning curve effect: success depends more on accumulated experience (flight number, block version, reuse count) "
    "than on payload alone — a competitor should expect SpaceX's cost advantage to keep growing.",
    "Test-set accuracy is identical for all models because the test set is only 18 launches; cross-validation accuracy "
    "is the more reliable criterion for model selection.",
    "Recommendation: retrain with newer launches (2021+) and add features such as booster reuse history and weather "
    "to reduce false positives.",
], size=16, gap=0.035)
save(fig)

fig = slide("Appendix")
bullets(fig, [
    f"GitHub repository (notebooks & Python files): {GITHUB}",
    "- 1_jupyter-labs-spacex-data-collection-api.ipynb, 2_jupyter-labs-webscraping.ipynb, 3_labs-jupyter-spacex-Data-wrangling.ipynb",
    "- jupyter-labs-eda-sql-coursera_sqllite.ipynb, edadataviz.ipynb, lab_jupyter_launch_site_location.ipynb",
    "- spacex_dash_app.py, SpaceX_Machine_Learning_Prediction_Part_5.ipynb",
    "Datasets: dataset_part_1/2/3.csv, Spacex.csv, spacex_launch_geo.csv, spacex_launch_dash.csv (IBM Skills Network)",
    "Tools: Python, pandas, NumPy, Matplotlib, Seaborn, SQLite, Folium, Plotly Dash, scikit-learn",
], size=14, gap=0.025)
save(fig)

fig = slide("", dark=True)
fig.text(0.5, 0.5, "Thank you!", fontsize=48, color="white", weight="bold", ha="center", va="center")
save(fig)
pdf.close()
print("pages:", page[0], "best:", best, {k: (round(v["cv"], 3), round(v["test"], 3)) for k, v in res.items()})
