# ll2sumo

`ll2sumo` converts Lanelet2 OSM maps into SUMO networks.

The converter is designed for vehicle road networks. It exports SUMO plain XML files, runs `netconvert`, and writes a final `network.net.xml` that can be opened by `sumo`, `sumo-gui`, and SUMO tools such as `randomTrips.py`.

## What It Does

- Projects Lanelet2 geo coordinates to UTM with PROJ, and infers the local coordinate offset of MGRS-style maps.
- Converts `road` lanelets into SUMO edges and lanes.
- Infers predecessor / successor topology from shared Lanelet2 boundary nodes.
- Handles `intersection_area` lanelets as intersection clusters.
- Exports SUMO connections and internal connection geometry.
- Converts vehicle traffic light relations into SUMO TLS-controlled junctions.
- Patches generated TLS phases with a Japanese-style static phase heuristic.
- Writes audit data for lane geometry, connection geometry, TLS conversion, and randomTrips-safe edge weights.

## Installation

The converter is Python plus SUMO. It has two Python dependencies:

- `pyproj`, which carries PROJ and performs the WGS84 to UTM projection of Lanelet2 geo coordinates,
- `sumolib`, SUMO's own pure Python helper library, used to locate the `netconvert` binary.

SUMO itself is a runtime dependency because `netconvert` is executed as a subprocess.

The `sumo` extra installs the official `eclipse-sumo` wheel, which ships the SUMO binaries (`netconvert`, `sumo`, `sumo-gui`, ...) and the SUMO `tools/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[sumo]"
```

This pins the SUMO version the converter is validated against:

```text
eclipse-sumo==1.26.0
```

If you already have SUMO installed on the host, install without the extra:

```bash
pip install -e .
```

Docker remains fully supported as an alternative to both, and does not require a host Python environment at all. See [Docker](#docker).

### Platform Notes

`eclipse-sumo` 1.26.0 publishes wheels for:

```text
macosx_14_0_arm64
manylinux_2_28_aarch64
manylinux_2_28_x86_64
win_amd64
```

On Apple Silicon this runs natively, which is faster than the official SUMO Docker image. That image is `linux/amd64` only, so on arm64 hosts Docker runs it through emulation.

There is no macOS x86_64 wheel, and no musl wheel. Use a host SUMO installation or Docker on those platforms.

### How netconvert Is Located

Binary lookup is delegated to `sumolib.checkBinary`, so it follows the usual SUMO conventions. In order:

1. the path given to `--netconvert-binary`,
2. the `NETCONVERT_BINARY` environment variable,
3. `SUMO_HOME/bin`,
4. the `eclipse-sumo` wheel installed in the current interpreter,
5. `PATH`.

So an activated virtualenv, an exported `SUMO_HOME`, and a system-wide SUMO installation all work without extra configuration. The binary that actually ran is recorded in `conversion.report.json` under `netconvert.binary`.

To check what a given environment resolves to:

```bash
python3 -c "import sumolib; print(sumolib.checkBinary('netconvert'))"
```

## Recommended Workflow

For most validation and simulation workflows, use:

```bash
--lane-change-mode unrestricted
```

This is recommended because Lanelet2-derived lane-change restrictions can make SUMO random traffic more likely to jam, especially when using `randomTrips.py` on dense urban maps. `unrestricted` omits SUMO `changeLeft` / `changeRight` restrictions and is currently the safer default choice for route generation, smoke tests, and visual inspection.

The code default is still `lanelet-infer` so the converter does not silently discard Lanelet2 lane-change semantics. For randomTrips checks on dense urban maps, pass `--lane-change-mode unrestricted` explicitly.

## Repository Layout

Keep local map data and generated files outside Git:

```text
lanelet2_to_sumo/
  pyproject.toml           # package metadata and the optional `sumo` extra
  ll2sumo/                 # converter source code
  tests/                   # unit tests
  map/                     # local input maps, not committed
    input.osm
  out/                     # generated SUMO outputs, not committed
    example-network/
  reference/               # optional local SUMO reference tree, not committed
```

The `map/`, `out/`, and `reference/` directories are ignored by Git and are also excluded from Docker image builds.

## Input Map Placement

Place the source Lanelet2 OSM map under `map/`:

```bash
mkdir -p map
cp /path/to/input.osm map/input.osm
```

Generated files are written under `out/`.

Both directories are mounted into the container by the Docker workflow; see [Input Map Placement For Docker](#input-map-placement-for-docker).

## Convert

This section uses a local Python environment. For the Docker equivalent, see [Convert With Docker](#convert-with-docker).

Recommended conversion:

```bash
mkdir -p out/example-network

python3 -m ll2sumo.convert \
  --input map/input.osm \
  --out-dir out/example-network \
  --lane-change-mode unrestricted
```

An `ll2sumo` console script is installed by `pip install`, so this is equivalent:

```bash
ll2sumo \
  --input map/input.osm \
  --out-dir out/example-network \
  --lane-change-mode unrestricted
```

To force a specific `netconvert` binary:

```bash
python3 -m ll2sumo.convert \
  --input map/input.osm \
  --out-dir out/example-network \
  --lane-change-mode unrestricted \
  --netconvert-binary /path/to/netconvert
```

Generated files:

```text
out/example-network/
  network.nod.xml
  network.edg.xml
  network.con.xml
  network.net.xml
  conversion.report.json
  retention.sidecar.json
  signal_id_mapping.json
  randomtrips.safe.src.xml
  randomtrips.safe.dst.xml
  randomtrips.safe.via.xml
```

`signal_id_mapping.json` is emitted for the default `--signal-mode jp-static`.
It contains both group-level Lanelet2 signal mappings and SUMO connection-level mappings:

- `lanelet_to_sumo`: Lanelet2 traffic-light regulatory elements and `refers` / `ref_line` way IDs mapped to the final SUMO `<tlLogic id>`.
- `sumo_link_to_lanelet_signal`: final SUMO `tlLogic id + linkIndex` records mapped back to Lanelet2 `refers` way IDs, including diagnostic fallback candidates.
- `lanelet_signal_to_sumo_links`: runtime synchronization lookup keyed by Lanelet2 `refers` way ID. It contains direct source-lanelet matches only; fallback candidates stay in `sumo_link_to_lanelet_signal` for diagnostics.

Open the generated network in SUMO GUI:

```bash
sumo-gui -n out/example-network/network.net.xml
```

## randomTrips Validation

Use the generated safe weights when running `randomTrips.py`.

The safe weights set source / destination / via weights to zero for disconnected or dead-end edges that should not be used for random route generation.

For the Docker equivalent, see [randomTrips Validation With Docker](#randomtrips-validation-with-docker).

`randomTrips.py` lives in the SUMO tools directory. With the `sumo` extra installed, resolve it from the wheel:

```bash
cd out/example-network

python3 "$(python3 -c 'import sumo, os; print(os.path.join(sumo.SUMO_HOME, "tools", "randomTrips.py"))')" \
  -n network.net.xml \
  --weights-prefix randomtrips.safe \
  --validate \
  -e 200 \
  -p 1 \
  -r test.rou.xml
```

With a host SUMO installation, if `SUMO_HOME` is exported:

```bash
cd out/example-network

python3 "$SUMO_HOME/tools/randomTrips.py" \
  -n network.net.xml \
  --weights-prefix randomtrips.safe \
  --validate \
  -e 200 \
  -p 1 \
  -r test.rou.xml
```

## Run SUMO

For the Docker equivalent, see [Run SUMO With Docker](#run-sumo-with-docker).

Headless SUMO:

```bash
cd out/example-network

sumo \
  -n network.net.xml \
  -r test.rou.xml \
  --duration-log.disable \
  --no-step-log true \
  --summary-output test.summary.xml \
  --tripinfo-output test.tripinfo.xml \
  --fcd-output test.fcd.xml
```

Visual inspection:

```bash
sumo-gui \
  -n out/example-network/network.net.xml \
  -r out/example-network/test.rou.xml
```

## CLI Options

Show all options:

```bash
python3 -m ll2sumo.convert --help
```

Main options:

- `--lane-change-mode lanelet-infer`
  - Keeps Lanelet2-inferred lane-change restrictions.
  - This is the code default because it preserves map semantics.

- `--lane-change-mode unrestricted`
  - Omits SUMO lane-change restriction attributes.
  - Recommended for randomTrips validation and general smoke testing.

- `--signal-mode jp-static`
  - Default signal mode.
  - Uses vehicle traffic light relations and lets `netconvert` build SUMO TLS-controlled junctions, then patches static phases with a Japanese-style heuristic.

- `--signal-mode none`
  - Disables signal export.

- `--skip-netconvert`
  - Writes SUMO plain XML files but does not build `network.net.xml`.
  - This path needs no SUMO installation at all.

- `--netconvert-binary /path/to/netconvert`
  - Uses a specific `netconvert` executable instead of the resolved one.
  - `NETCONVERT_BINARY` does the same thing through the environment.

## Reports

`conversion.report.json` contains conversion and audit summaries, including:

- lanelet / edge / connection counts
- lane-change summary
- signal conversion summary
- signal ID mapping summary
- TLS phase patch summary
- lane length vs shape patch summary
- internal lane shape audit
- internal lane repair summary
- connection shape summary
- connectivity summary

Check these fields first when validating a generated network:

```text
internal_shape_audit.degenerate_internal_lane_count
internal_shape_repair.repaired_internal_lane_count
connection_shape_summary.unshaped_connection_count
connectivity_summary
```

## Docker

The Python setup above is the default path. Docker remains fully supported, and is the way to run the converter without depending on a host Python environment or a host SUMO installation.

Use it when:

- the host platform has no `eclipse-sumo` wheel (macOS x86_64, musl-based Linux),
- you want the exact OS-level SUMO build used for converter validation, rather than the wheel build,
- you want one pinned image for CI or for sharing a reproducible environment.

### Docker Setup

Build the image:

```bash
docker build --platform linux/amd64 -t ll2sumo:latest .
```

The image installs the converter with `pip`, so it carries the same Python dependencies as a local install, and takes `netconvert` and the SUMO tools from the base image.

The Docker image is based on:

```text
ghcr.io/eclipse-sumo/sumo:v1_26_0
```

This matches the SUMO version used for converter validation:

```text
SUMO netconvert 1.26.0
```

The official SUMO `v1_26_0` image is `linux/amd64`. On Apple Silicon or other arm64 hosts, Docker runs it through emulation, so conversion is slower than native execution, including the native arm64 `eclipse-sumo` wheel.

### Input Map Placement For Docker

The Docker command mounts `map/` read-only at `/data/input`:

```text
host:      ./map/input.osm
container: /data/input/input.osm
```

Generated files are written under `out/`, mounted in the container as `/data/out`:

```text
host output:      ./out/example-network/
container output: /data/out/example-network/
```

### Convert With Docker

Recommended conversion:

```bash
mkdir -p out/example-network

docker run --rm \
  --platform linux/amd64 \
  -v "$PWD/map:/data/input:ro" \
  -v "$PWD/out:/data/out" \
  ll2sumo:latest \
  --input /data/input/input.osm \
  --out-dir /data/out/example-network \
  --lane-change-mode unrestricted
```

The generated files and reports are the same as for the local Python workflow; see [Convert](#convert) and [Reports](#reports).

### randomTrips Validation With Docker

```bash
docker run --rm \
  --platform linux/amd64 \
  -v "$PWD/out:/data/out" \
  --entrypoint python3 \
  ll2sumo:latest \
  /usr/share/sumo/tools/randomTrips.py \
  -n /data/out/example-network/network.net.xml \
  --weights-prefix /data/out/example-network/randomtrips.safe \
  --validate \
  -e 200 \
  -p 1 \
  -r /data/out/example-network/test.rou.xml
```

### Run SUMO With Docker

Headless SUMO in Docker:

```bash
docker run --rm \
  --platform linux/amd64 \
  -v "$PWD/out:/data/out" \
  --entrypoint sumo \
  ll2sumo:latest \
  -n /data/out/example-network/network.net.xml \
  -r /data/out/example-network/test.rou.xml \
  --duration-log.disable \
  --no-step-log true \
  --summary-output /data/out/example-network/test.summary.xml \
  --tripinfo-output /data/out/example-network/test.tripinfo.xml \
  --fcd-output /data/out/example-network/test.fcd.xml
```

Visual inspection on the host:

```bash
sumo-gui \
  -n out/example-network/network.net.xml \
  -r out/example-network/test.rou.xml
```

## SUMO Build Differences

The `eclipse-sumo` wheel, the Docker image, and a host SUMO installation may all report SUMO `1.26.0`, but `.net.xml` output can still differ at the byte level when the builds differ by OS, CPU architecture, compiler, or packaged libraries.

Observed differences are usually small coordinate / angle rounding changes and occasional internal edge numbering differences. Validate the generated network by behavior and report fields, not by byte-for-byte equality across different SUMO builds.

## Tests

Run unit tests:

```bash
python3 -m unittest discover -s tests -v
```

The unit tests do not invoke `netconvert`, so they run without a SUMO installation. They do need `pyproj` and `sumolib`, which `pip install -e .` provides.

## Current Limitations

- The converter mainly targets vehicle `road` lanelets.
- Pedestrian routing from `crosswalk` lanelets is not fully exported yet.
- Traffic signal timing is inferred; Lanelet2 usually does not provide full signal programs.
- `randomTrips.py` can still expose collisions or teleports depending on generated random demand and inferred signal behavior.
- The converter does not automatically connect Lanelet2 components that are geometrically close but not topologically connected in the input map.
