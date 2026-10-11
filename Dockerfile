# ImputeForge — reproducible CPU-only image
# Author: 晨星
FROM python:3.13-slim

LABEL org.opencontainers.image.title="ImputeForge" \
      org.opencontainers.image.description="EM-anchored MICE with per-column conditional-model selection and probe-calibrated MNAR correction" \
      org.opencontainers.image.authors="晨星" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY requirements.lock.txt ./requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md cli.py ./
COPY imputeforge ./imputeforge
COPY tests ./tests
COPY examples ./examples

# Import smoke test: the package must load with zero third-party surprises.
RUN python -c "import imputeforge, numpy; print('imputeforge', imputeforge.__version__, '| numpy', numpy.__version__)"

ENTRYPOINT ["python", "cli.py"]
CMD ["list"]
