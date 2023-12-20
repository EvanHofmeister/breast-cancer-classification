FROM python:3.10-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

# Set working directory
WORKDIR /app

# Copy pipfile requirements
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install project dependencies
RUN pipenv install --deploy --system && \
rm -rf /root/.cache

# Expose Port
EXPOSE 5000

# Run Gunicorn/Flask
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "flask_app.predict:app"]
