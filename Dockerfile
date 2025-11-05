# Base with micromamba for fast conda env
FROM mambaorg/micromamba:1.5.8

WORKDIR /app

# Copy environment and create env
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n bh_ea -f /tmp/environment.yml &&     micromamba clean --all --yes
SHELL ["/bin/bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/envs/bh_ea/bin:$PATH

# Copy app after env to leverage layer cache
COPY . /app

# Expose for Cloud Run
ENV PORT=8080
CMD exec gunicorn -b 0.0.0.0:$PORT app:app
