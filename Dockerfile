# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy the project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install dependencies using uv
# Syncing with --no-dev to install only production dependencies
RUN uv sync --no-dev --frozen

# Collect static files
RUN uv run python manage.py collectstatic --noinput

# Expose the port
EXPOSE 8080

# Command to run the application using gunicorn
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8080", "portfolio_core.wsgi:application"]
