ARG SUMO_IMAGE=ghcr.io/eclipse-sumo/sumo:v1_26_0
FROM ${SUMO_IMAGE}

ENV PYTHONUNBUFFERED=1
ENV SUMO_HOME=/usr/share/sumo
# The official SUMO v1_26_0 image stores Arrow/Parquet runtime libraries here.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/pyarrow

WORKDIR /workspace

# Install the converter the same way the README documents, so its Python
# dependencies (pyproj, sumolib) come from the package metadata rather than
# from assumptions about what the base image happens to ship. netconvert and
# the SUMO tools come from the image, so the `sumo` extra is not installed.
COPY pyproject.toml README.md /opt/ll2sumo/
COPY ll2sumo /opt/ll2sumo/ll2sumo
RUN pip3 install --no-cache-dir /opt/ll2sumo

ENTRYPOINT ["python3", "-m", "ll2sumo.convert"]
