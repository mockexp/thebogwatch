# pipeline/get_infrastructure.py
# Step D1: downloads wind turbines, roads/tracks and forest for a site from OpenStreetMap,
# saves them as GeoJSON (for the distance bands and the 3D viewer), and draws a check map.
# Data: (c) OpenStreetMap contributors, ODbL.
# Usage:
#     python pipeline\get_infrastructure.py meenbog

import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
from shapely import wkt as shapely_wkt

from config1 import SITES, WORK_DIR
from coherence_check1 import MARKERS

ox.settings.use_cache = True
ox.settings.cache_folder = str(WORK_DIR / "osm_cache")   # reruns reuse earlier downloads

KEEP_COLUMNS = ["id", "name", "ref", "highway", "tracktype", "surface",
                "generator:source", "generator:output:electricity", "landuse", "natural"]
ROAD_GROUPS = {   # simplify OpenStreetMap's many road types into a few groups for the map
    "track": "track (access/forestry)", "service": "service road",
    "path": "path", "footway": "path", "bridleway": "path", "cycleway": "path",
}


def fetch(polygon, tags, label):
    """1. Ask OpenStreetMap for features matching the tags inside the polygon."""
    try:
        found = ox.features_from_polygon(polygon, tags)
    except Exception as error:  # osmnx raises an error when nothing matches (or the service fails)
        print(f"  {label}: none returned ({type(error).__name__}: {str(error)[:80]})")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    found = found.reset_index()
    print(f"  {label}: {len(found)} features returned")
    return found


def tidy(gdf):
    """Keep only useful columns, so the saved files stay small."""
    columns = [c for c in KEEP_COLUMNS if c in gdf.columns]
    return gdf[columns + ["geometry"]].copy()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SITES:
        print("Usage: python pipeline\\get_infrastructure.py <site>")
        return
    site_id = sys.argv[1]
    box = shapely_wkt.loads(SITES[site_id]["wkt"])
    print(f"Asking OpenStreetMap about {SITES[site_id]['name']}...")

    # 1-2. Wind turbines, as points.
    generators = fetch(box, {"power": "generator"}, "power generators")
    if len(generators) and "generator:source" in generators.columns:
        turbines = generators[generators["generator:source"] == "wind"].copy()
    else:
        turbines = generators.iloc[0:0].copy()
    utm = gpd.GeoSeries([box], crs="EPSG:4326").estimate_utm_crs()
    if len(turbines):
        turbines = turbines.to_crs(utm)
        turbines["geometry"] = turbines.geometry.centroid
        turbines = turbines.to_crs("EPSG:4326")

    # 1-2. Roads and tracks, as lines cut to the box.
    roads = fetch(box, {"highway": True}, "roads and tracks")
    if len(roads):
        roads = roads[roads.geom_type.isin(["LineString", "MultiLineString"])]
        roads = gpd.clip(roads, box)

    # 1-2. Forest, as areas cut to the box.
    forest = fetch(box, {"landuse": "forest", "natural": "wood"}, "forest and woodland")
    if len(forest):
        forest = forest[forest.geom_type.isin(["Polygon", "MultiPolygon"])]
        forest = gpd.clip(forest, box)

    # 3. Measurements.
    print(f"\nWind turbines found: {len(turbines)}")
    if len(turbines):
        names = turbines["name"].dropna().unique() if "name" in turbines.columns else []
        if len(names):
            print(f"  Names given in OpenStreetMap: {', '.join(map(str, names[:10]))}")
    if len(roads):
        lengths = roads.to_crs(utm).length / 1000
        by_type = lengths.groupby(roads["highway"]).sum().sort_values(ascending=False)
        print(f"Roads and tracks: {lengths.sum():.1f} km in total")
        for kind, km in by_type.items():
            print(f"  {kind:15s} {km:6.1f} km")
    else:
        print("Roads and tracks: none")
    if len(forest):
        print(f"Forest and woodland: {forest.to_crs(utm).area.sum() / 1e6:.1f} km²")
    else:
        print("Forest and woodland: none mapped")

    # 4. Save the map files.
    for gdf, name in [(turbines, "turbines"), (roads, "roads"), (forest, "forest")]:
        path = WORK_DIR / f"{site_id}_{name}.geojson"
        if len(gdf):
            tidy(gdf).to_file(path, driver="GeoJSON")
            print(f"Saved {path.name}")
        else:
            print(f"Nothing to save for {name}")

    # 5. Check map, in metres.
    fig, ax = plt.subplots(figsize=(11, 10))
    gpd.GeoSeries([box], crs="EPSG:4326").to_crs(utm).boundary.plot(ax=ax, color="black", linewidth=1)
    if len(forest):
        forest.to_crs(utm).plot(ax=ax, color="forestgreen", alpha=0.3, label="forest")
    if len(roads):
        grouped = roads["highway"].map(lambda h: ROAD_GROUPS.get(h, "road"))
        colours = {"road": "black", "service road": "dimgray", "track (access/forestry)": "saddlebrown", "path": "orange"}
        for group, colour in colours.items():
            part = roads[grouped == group]
            if len(part):
                part.to_crs(utm).plot(ax=ax, color=colour, linewidth=1.2, label=group)
    if len(turbines):
        turbines.to_crs(utm).plot(ax=ax, color="red", marker="^", markersize=60, label="wind turbine")
    slide = gpd.GeoSeries(gpd.points_from_xy([MARKERS[site_id][0]], [MARKERS[site_id][1]]), crs="EPSG:4326").to_crs(utm)
    ax.plot(slide.x, slide.y, marker="x", color="black", markersize=12, mew=3, label="2020 slide")
    ax.set_title(f"{SITES[site_id]['name']}: OpenStreetMap infrastructure\n(c) OpenStreetMap contributors")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(WORK_DIR / f"{site_id}_infrastructure.png", dpi=110)
    plt.close(fig)
    print(f"Saved picture: {site_id}_infrastructure.png")


if __name__ == "__main__":
    main()