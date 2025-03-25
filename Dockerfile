FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy necessary project components
COPY ./remoterl/remote_simulator.py remoterl/
COPY ./remoterl/remote_env.py remoterl/
COPY ./remoterl/server remoterl/server/
COPY ./remoterl/utils remoterl/utils/

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set entry pointg
CMD ["python", "-m", "remoterl.remote_simulator"]
