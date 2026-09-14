#!/usr/bin/env python3
"""
route_tiles.py - list (and optionally build) the Ortho4XP tiles a flight route crosses.

Input can be one of:
  --simbrief USERNAME        latest SimBrief OFP for that username (or --simbrief-id PILOT_ID)
  --lnmpln plan.lnmpln       Little Navmap flight plan
  --pln plan.pln             FSX / Prepar3D flight plan (Little Navmap can export these)
  --route KLAX KLAS ...      ICAO codes and/or "lat,lon" points, resolved via OurAirports data

What it does:
  1. Interpolates the great-circle track between the waypoints.
  2. Collects every 1x1 degree tile within --corridor nautical miles of the track.
  3. Marks tiles within --airport-radius nm of the origin / destination for --airport-zl,
     everything else for --zl.
  4. Prints the list with a rough size estimate. With --build it runs the Ortho4XP FSX/P3D
     batch build for those tiles (same steps as the GUI "Batch Build" with "Build For ESP" ticked).

Examples:
  python route_tiles.py --route KLAX KLAS
  python route_tiles.py --simbrief myname --zl 15 --airport-zl 16
  python route_tiles.py --lnmpln "C:\\Users\\me\\Documents\\Little Navmap\\KLAX-KLAS.lnmpln" --build

Must be run from (or live in) the Ortho4XP folder, next to Ortho4XP_v130.py.
"""
import argparse
import csv
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

# numpy's OpenBLAS reserves a buffer per logical CPU at import; on a 32-thread machine under
# memory pressure that fails with "OpenBLAS error: Memory allocation still failed". Ortho4XP
# does no heavy BLAS work, so cap it before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

NM_PER_DEG = 60.0
EARTH_R_NM = 3440.065
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# rough finished size per tile in GB, from typical Bing land tiles (P3D output + imagery cache)
SIZE_GB = {14: 0.3, 15: 1.0, 16: 4.0, 17: 15.0, 18: 60.0}


# ----------------------------------------------------------------------------- geometry
def gc_distance_nm(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_NM * math.asin(math.sqrt(a))


def gc_interpolate(lat1, lon1, lat2, lon2, f):
    """Point at fraction f (0..1) along the great circle from 1 to 2."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2
                                + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    if d < 1e-12:
        return lat1, lon1
    a = math.sin((1 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))


def destination(lat, lon, bearing_deg, dist_nm):
    p1, l1, b = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    dr = dist_nm / EARTH_R_NM
    p2 = math.asin(math.sin(p1) * math.cos(dr) + math.cos(p1) * math.sin(dr) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(dr) * math.cos(p1), math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def tile_of(lat, lon):
    return int(math.floor(lat)), int(math.floor(lon))


def tile_name(lat, lon):
    return "%+03d%+04d" % (lat, lon)


# ----------------------------------------------------------------------------- inputs
def parse_dms(s):
    """FSX/P3D WorldPosition like N33° 56' 33.00",W118° 24' 28.80",+000126.00"""
    m = re.findall(r"([NSEW])\s*(\d+)°\s*(\d+)'\s*([\d.]+)\"", s)
    out = {}
    for hemi, d, mi, se in m:
        v = int(d) + int(mi) / 60 + float(se) / 3600
        if hemi in "SW":
            v = -v
        out["lat" if hemi in "NS" else "lon"] = v
    return out["lat"], out["lon"]


def read_pln(path):
    root = ET.parse(path).getroot()
    pts = []
    for wp in root.iter("ATCWaypoint"):
        wpos = wp.find("WorldPosition")
        if wpos is not None and wpos.text:
            pts.append(parse_dms(wpos.text))
    return pts


def read_lnmpln(path):
    root = ET.parse(path).getroot()
    pts = []
    for pos in root.iter("Pos"):
        if "Lat" in pos.attrib and "Lon" in pos.attrib:
            pts.append((float(pos.attrib["Lat"]), float(pos.attrib["Lon"])))
    return pts


def read_simbrief(username=None, userid=None):
    import requests
    params = {"json": 1}
    if username:
        params["username"] = username
    else:
        params["userid"] = userid
    r = requests.get("https://www.simbrief.com/api/xml.fetcher.php", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "fetch" in data and data["fetch"].get("status") not in (None, "Success"):
        raise SystemExit("SimBrief: " + str(data["fetch"].get("status")))
    pts = [(float(data["origin"]["pos_lat"]), float(data["origin"]["pos_long"]))]
    fixes = data.get("navlog", {}).get("fix", [])
    if isinstance(fixes, dict):
        fixes = [fixes]
    for fx in fixes:
        pts.append((float(fx["pos_lat"]), float(fx["pos_long"])))
    pts.append((float(data["destination"]["pos_lat"]), float(data["destination"]["pos_long"])))
    label = "%s-%s" % (data["origin"]["icao_code"], data["destination"]["icao_code"])
    return pts, label


def airport_lookup(codes, cache_dir):
    """Resolve ICAO codes with the OurAirports CSV (downloaded once and cached)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "airports.csv")
    if not os.path.isfile(cache):
        import requests
        print("Downloading airport database (once) ...")
        r = requests.get(OURAIRPORTS_URL, timeout=120)
        r.raise_for_status()
        with open(cache, "wb") as f:
            f.write(r.content)
    want = {c.upper() for c in codes}
    found = {}
    with open(cache, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in (row.get("ident", ""), row.get("gps_code", ""), row.get("icao_code", "")):
                if key and key.upper() in want and key.upper() not in found:
                    found[key.upper()] = (float(row["latitude_deg"]), float(row["longitude_deg"]))
    missing = want - set(found)
    if missing:
        raise SystemExit("Unknown airport code(s): " + ", ".join(sorted(missing)))
    return found


def read_route_tokens(tokens, cache_dir):
    codes = [t for t in tokens if "," not in t]
    lookup = airport_lookup(codes, cache_dir) if codes else {}
    pts = []
    for t in tokens:
        if "," in t:
            a, b = t.split(",", 1)
            pts.append((float(a), float(b)))
        else:
            pts.append(lookup[t.upper()])
    return pts


# ----------------------------------------------------------------------------- tiles
def tiles_for_route(points, corridor_nm, step_nm=1.0):
    tiles = set()
    offsets = [0.0]
    o = 2.0
    while o <= corridor_nm:
        offsets += [o, -o]
        o += 2.0
    if corridor_nm > 0 and corridor_nm not in offsets:
        offsets += [corridor_nm, -corridor_nm]
    for (a, b) in zip(points[:-1], points[1:]):
        dist = gc_distance_nm(*a, *b)
        n = max(1, int(dist / step_nm))
        for i in range(n + 1):
            lat, lon = gc_interpolate(*a, *b, i / n)
            brg = bearing(*a, *b)
            for off in offsets:
                if off == 0:
                    tiles.add(tile_of(lat, lon))
                else:
                    tiles.add(tile_of(*destination(lat, lon, brg + (90 if off > 0 else -90), abs(off))))
    return tiles


def tiles_near(lat, lon, radius_nm):
    tiles = {tile_of(lat, lon)}
    r = 2.0
    while r <= radius_nm:
        for brg in range(0, 360, 15):
            tiles.add(tile_of(*destination(lat, lon, brg, r)))
        r += 2.0
    for brg in range(0, 360, 10):
        tiles.add(tile_of(*destination(lat, lon, brg, radius_nm)))
    return tiles


# ----------------------------------------------------------------------------- build
def run_build(plan, provider):
    """plan: list of (lat, lon, zl). Runs the same steps as the GUI batch build with Build For ESP."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    sys.path.insert(0, os.path.join(here, "src"))
    import O4_File_Names as FNAMES
    sys.path.append(FNAMES.Provider_dir)
    import O4_Imagery_Utils as IMG
    import O4_Tile_Utils as TILE
    import O4_ESP_Globals
    import O4_UI_Utils as UI
    import O4_Config_Utils as CFG  # last, it may modify other modules' variables

    for directory in (FNAMES.Preview_dir, FNAMES.Provider_dir, FNAMES.Extent_dir, FNAMES.Filter_dir, FNAMES.OSM_dir,
                      FNAMES.Mask_dir, FNAMES.Imagery_dir, FNAMES.Elevation_dir, FNAMES.Geotiff_dir, FNAMES.Patch_dir,
                      FNAMES.Tile_dir, FNAMES.Tmp_dir):
        os.makedirs(directory, exist_ok=True)
    IMG.initialize_extents_dict()
    IMG.initialize_color_filters_dict()
    IMG.initialize_providers_dict()
    IMG.initialize_combined_providers_dict()

    if not CFG.ESP_resample_loc or not os.path.isfile(CFG.ESP_resample_loc):
        raise SystemExit("ESP_resample_loc in Ortho4XP.cfg is empty or wrong: %r" % CFG.ESP_resample_loc)

    O4_ESP_Globals.build_for_ESP = True
    O4_ESP_Globals.do_build_masks = True

    # one batch per zoom level: the FSX/P3D path needs mask_zl == tile zl
    by_zl = {}
    for lat, lon, zl in plan:
        by_zl.setdefault(zl, []).append((lat, lon))
    for zl in sorted(by_zl, reverse=True):
        lst = sorted(by_zl[zl])
        CFG.default_website = provider
        CFG.default_zl = zl
        CFG.mask_zl = zl
        lat0, lon0 = lst[0]
        tile = CFG.Tile(lat0, lon0, "")
        tile.default_website = provider
        tile.default_zl = zl
        tile.mask_zl = zl
        print("\n===== ZL%d batch: %s =====\n" % (zl, " ".join(tile_name(*t) for t in lst)))
        UI.is_working = False
        ok = TILE.build_tile_list(tile, lst, True, True, True, True, False, False)
        if not ok:
            print("Batch at ZL%d stopped early (see messages above)." % zl)
            return 1
    return 0


# ----------------------------------------------------------------------------- P3D registration
def documents_folder():
    """The user's Documents folder, honouring OneDrive / known-folder redirection on Windows."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
                return buf.value
        except Exception:
            pass
    return os.path.join(os.path.expanduser("~"), "Documents")


def write_p3d_addon(name, plan, provider, here):
    """Write <Documents>\\Prepar3D vX Add-ons\\<name>\\add-on.xml pointing at every built tile of the plan.
    Prepar3D discovers it on next start and asks once whether to activate it. No files are copied."""
    from xml.sax.saxutils import escape
    docs = documents_folder()
    addons_root = None
    for v in ("6", "5", "4"):
        cand = os.path.join(docs, "Prepar3D v%s Add-ons" % v)
        if os.path.isdir(cand):
            addons_root = cand
            break
    if not addons_root:
        raise SystemExit("No 'Prepar3D vX Add-ons' folder under %s. Start Prepar3D once so it creates it." % docs)
    comps = []
    for lat, lon, zl in plan:
        band = "%+03d%+04d" % ((lat // 10) * 10, (lon // 10) * 10)
        scen = os.path.join(here, "Orthophotos", band, tile_name(lat, lon), "%s_%d" % (provider, zl), "ADDON_SCENERY")
        if os.path.isdir(os.path.join(scen, "scenery")) and os.listdir(os.path.join(scen, "scenery")):
            comps.append((tile_name(lat, lon), zl, os.path.abspath(scen)))
    if not comps:
        raise SystemExit("None of the planned tiles has BGL output yet; build first.")
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<SimBase.Document Type="AddOnXml" version="4,0" id="add-on">',
             '  <AddOn.Name>%s</AddOn.Name>' % escape(name),
             '  <AddOn.Description>Ortho4XP photo scenery, %d tiles, generated by route_tiles.py</AddOn.Description>' % len(comps)]
    for tname, zl, path in comps:
        lines += ['  <AddOn.Component>',
                  '    <Category>Scenery</Category>',
                  '    <Path>%s</Path>' % escape(path),
                  '    <Name>Ortho %s ZL%d</Name>' % (tname, zl),
                  '  </AddOn.Component>']
    lines.append('</SimBase.Document>')
    out_dir = os.path.join(addons_root, name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "add-on.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote %s with %d scenery entries. Start Prepar3D and accept the new add-on when asked." % (out, len(comps)))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--simbrief", metavar="USERNAME")
    src.add_argument("--simbrief-id", metavar="PILOT_ID")
    src.add_argument("--lnmpln", metavar="FILE")
    src.add_argument("--pln", metavar="FILE")
    src.add_argument("--route", nargs="+", metavar="ICAO|lat,lon")
    ap.add_argument("--corridor", type=float, default=15, help="nm each side of track (default 15)")
    ap.add_argument("--zl", type=int, default=15, help="zoom level for en-route tiles (default 15)")
    ap.add_argument("--airport-zl", type=int, default=16, help="zoom level near origin/destination (default 16)")
    ap.add_argument("--airport-radius", type=float, default=20, help="nm around origin/destination (default 20)")
    ap.add_argument("--provider", default="BI", help="imagery provider code (default BI = Bing)")
    ap.add_argument("--build", action="store_true", help="run the Ortho4XP FSX/P3D batch build for these tiles")
    ap.add_argument("--out", metavar="FILE", help="also write the tile list to this file")
    ap.add_argument("--skip-done", action="store_true",
                    help="leave out tiles that already have BGL output for this provider and zoom level")
    ap.add_argument("--p3d-addon", metavar="NAME",
                    help="after listing/building, register the built tiles in Prepar3D by writing "
                         "<Documents>\\Prepar3D vX Add-ons\\NAME\\add-on.xml (no files are copied)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, "tmp")
    label = "route"
    if args.simbrief or args.simbrief_id:
        points, label = read_simbrief(args.simbrief, args.simbrief_id)
    elif args.lnmpln:
        points = read_lnmpln(args.lnmpln)
        label = os.path.splitext(os.path.basename(args.lnmpln))[0]
    elif args.pln:
        points = read_pln(args.pln)
        label = os.path.splitext(os.path.basename(args.pln))[0]
    else:
        points = read_route_tokens(args.route, cache_dir)
        label = "-".join(args.route)
    if len(points) < 2:
        raise SystemExit("Need at least two waypoints, got %d" % len(points))

    total_nm = sum(gc_distance_nm(*a, *b) for a, b in zip(points[:-1], points[1:]))
    route_tiles = tiles_for_route(points, args.corridor)
    apt_tiles = tiles_near(*points[0], args.airport_radius) | tiles_near(*points[-1], args.airport_radius)

    plan = []
    for (lat, lon) in sorted(route_tiles | apt_tiles):
        zl = args.airport_zl if (lat, lon) in apt_tiles else args.zl
        plan.append((lat, lon, zl))

    full_plan = list(plan)
    if args.skip_done:
        import glob
        kept = []
        for lat, lon, zl in plan:
            band = "%+03d%+04d" % ((lat // 10) * 10, (lon // 10) * 10)
            tile_dir = os.path.join(here, "Orthophotos", band, tile_name(lat, lon), "%s_%d" % (args.provider, zl))
            n_inf = len(glob.glob(os.path.join(tile_dir, "*.inf")))
            n_bgl = len(glob.glob(os.path.join(tile_dir, "ADDON_SCENERY", "scenery", "*.bgl")))
            if n_inf and n_bgl >= n_inf:
                print("skipping %s ZL%d, already built (%d BGL)" % (tile_name(lat, lon), zl, n_bgl))
            else:
                if n_bgl:
                    print("rebuilding %s ZL%d, only %d of %d textures resampled" % (tile_name(lat, lon), zl, n_bgl, n_inf))
                kept.append((lat, lon, zl))
        plan = kept
        if not plan:
            print("Nothing left to build.")
            if args.p3d_addon:
                write_p3d_addon(args.p3d_addon, full_plan, args.provider, here)
            return 0

    est = sum(SIZE_GB.get(zl, 4.0) for _, _, zl in plan)
    print("Route %s: %d waypoints, %.0f nm, corridor %.0f nm" % (label, len(points), total_nm, args.corridor))
    print("%-9s %-4s %s" % ("tile", "ZL", "why"))
    for lat, lon, zl in plan:
        why = "airport area" if (lat, lon) in apt_tiles else "en route"
        print("%-9s %-4d %s" % (tile_name(lat, lon), zl, why))
    print("%d tiles, roughly %.0f GB of output, provider %s" % (len(plan), est, args.provider))

    if args.out:
        with open(args.out, "w") as f:
            for lat, lon, zl in plan:
                f.write("%d %d %d\n" % (lat, lon, zl))
        print("List written to", args.out)

    rc = 0
    if args.build:
        rc = run_build(plan, args.provider)
    else:
        print("\nDry run only. Add --build to start the Ortho4XP FSX/P3D batch build.")
    if args.p3d_addon:
        # register everything on the route that has output, including tiles skipped as already built
        write_p3d_addon(args.p3d_addon, full_plan, args.provider, here)
    return rc


if __name__ == "__main__":
    sys.exit(main())
