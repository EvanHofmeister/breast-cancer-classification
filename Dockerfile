FROM python:3.10-slim

ENV PYTHONUNBUFFERED=TRUE

ENV RUNNING_IN_DOCKER=true

RUN pip --no-cache-dir install pipenv

# Set working directory
WORKDIR /app

# Copy pipfile requirements
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install project dependencies
RUN pipenv install --deploy --system && \
rm -rf /root/.cache

# Copy the necessary folders and files
COPY ["flask_app/predict.py", "./src/flask_app/"]
COPY ["data", "./data/"]
COPY ["model", "./model/"]

# Expose Port
EXPOSE 5000

# Run Gunicorn/Flask
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "src.flask_app.predict:app"]
