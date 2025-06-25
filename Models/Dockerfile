# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including build tools for Stockfish)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*

# Download and compile Stockfish
RUN wget https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_16.tar.gz && \
    tar -xzf sf_16.tar.gz && \
    cd Stockfish-sf_16/src && \
    make -j $(nproc) build ARCH=x86-64-avx2 && \
    mv stockfish /usr/local/bin/ && \
    cd / && \
    rm -rf Stockfish-sf_16 sf_16.tar.gz

# Copy your application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available
EXPOSE 5000

# Set environment variable for Stockfish path
ENV STOCKFISH_PATH=/usr/local/bin/stockfish

# Run the application
CMD ["python", "app.py"]
