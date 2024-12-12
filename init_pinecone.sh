# init_pinecone.sh
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Initialize vectors
python3 << EOF
from exercise_rag import ExerciseRAG
import json

with open('exercise_data.json', 'r') as f:
    data = json.load(f)

rag = ExerciseRAG(data)
rag.initialize_vectors()
EOF