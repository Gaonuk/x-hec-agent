FROM python:3.11


WORKDIR /src
# Copy all source code
COPY . .

RUN pip install -r requirements.txt

# Run the server on port 80
CMD ["python", "main.py"]