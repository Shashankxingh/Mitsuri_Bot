# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory content into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (you can replace with your own secrets)
ENV GEMINI_API_KEY=your-gemini-api-key
ENV TELEGRAM_BOT_TOKEN=your-telegram-bot-token

# Expose port 80 (optional, only if needed for webhooks)
EXPOSE 80

# Command to run the bot
CMD ["python", "mitsuri.py"]
