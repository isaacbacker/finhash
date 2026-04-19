FROM python:3.11-slim

WORKDIR /app

# Copy package definition first for layer caching — dependencies
# are only re-installed when pyproject.toml changes, not on every
# code edit.
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

EXPOSE 8945

# Bind to 0.0.0.0 so the server is accessible outside the container.
CMD ["uvicorn", "finhash.api:app", "--host", "0.0.0.0", "--port", "8945"]
