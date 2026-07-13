FROM python:3.10

# Install system dependencies (e.g. libsndfile for Whisper/Soundfile)
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy files required to install python package in editable mode (-e .)
COPY ./requirements.txt /code/requirements.txt
COPY ./setup.py /code/setup.py
COPY ./src /code/src

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Run Uvicorn on default Hugging Face Spaces port (7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
