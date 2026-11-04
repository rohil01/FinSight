FROM python:3.12-slim

WORKDIR /home/appuser

RUN useradd --create-home --uid 1000 appuser

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && chown -R appuser:appuser /home/appuser

USER appuser

ENTRYPOINT ["finsight"]
CMD ["--help"]
