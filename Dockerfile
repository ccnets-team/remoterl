# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy necessary project components
COPY ./remoterl/remote_simulator.py remoterl/
COPY ./remoterl/remote_env.py remoterl/
COPY ./remoterl/server remoterl/server/
COPY ./remoterl/utils remoterl/utils/

# Install Python dependencies directly (hardcoded)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        websocket-client \
        msgpack \
        requests \
        typer \
        "gymnasium[mujoco]>=1.0.0"

# Expose port if needed (e.g., 8080)
EXPOSE 8080 

# ENTRYPOINT specifies the executable that runs
ENTRYPOINT ["python", "-m", "remoterl.remote_simulator"]

# CMD provides default arguments (can be overridden by user)
CMD ["--region", "ap-northeast-2", "--env", "Humanoid-v5", "--num_env_runners", "4"]