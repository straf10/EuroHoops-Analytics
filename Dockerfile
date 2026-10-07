# Local run of the read-only API (not deployed publicly): `docker compose up`, see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    EUROHOOPS_ROOT=/srv/eurohoops

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# The runtime dependencies from the lock file; uvicorn is a dev-only dependency of the project,
# so the image adds the locked version on its own.
RUN uv sync --frozen --no-dev --no-editable \
    && uv pip install --python /opt/venv/bin/python "uvicorn==0.54.0"

# The data (marts, raw caches, reports, logs) is mounted here, read-only, by docker-compose.yml.
WORKDIR /srv/eurohoops
EXPOSE 8000
CMD ["uvicorn", "--factory", "eurohoops.api.app:local_app", "--host", "0.0.0.0", "--port", "8000"]
