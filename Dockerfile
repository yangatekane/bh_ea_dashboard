# ---------------------------------------------------------
# 🧱 Base Image: Micromamba (fast conda-based build)
# ---------------------------------------------------------
FROM mambaorg/micromamba:1.5.8

# Set working directory
WORKDIR /app

# ---------------------------------------------------------
# 📦 Environment Setup
# ---------------------------------------------------------
# Copy environment definition and create environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n bh_ea -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Add micromamba env to PATH
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/envs/bh_ea/bin:$PATH
SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------------------
# 📁 Copy Application Code
# ---------------------------------------------------------
COPY . /app

# ---------------------------------------------------------
# 🌐 Expose Port for Cloud Run
# ---------------------------------------------------------
EXPOSE 8080
ENV PORT=8080

# ---------------------------------------------------------
# 🚀 Start Flask app with Gunicorn (inside micromamba env)
# ---------------------------------------------------------
CMD ["bash", "-lc", "micromamba run -n bh_ea gunicorn -b 0.0.0.0:$PORT app:app"]
