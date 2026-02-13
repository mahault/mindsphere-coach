FROM python:3.12-slim

# HF Spaces requires non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (cache layer)
COPY --chown=user pyproject.toml .
COPY --chown=user src/ src/
RUN pip install --no-cache-dir -e .

# Copy remaining source
COPY --chown=user . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "mindsphere.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
