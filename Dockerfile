# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app will run on
EXPOSE 10000

# Set environment variables (Render will override with actual values)
ENV PORT=10000

# Start the bot
CMD ["python", "mitsuri.py"]
