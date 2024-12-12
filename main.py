import os
from pydantic import BaseModel
from openai import OpenAI
from fastapi import FastAPI, Depends, HTTPException
from dotenv import load_dotenv
import phospho
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
import json
from typing import List, Optional, Dict, Any
from exercise_rag import ExerciseRAG  
from groq import Groq

load_dotenv()

client = OpenAI()
app = FastAPI()

# Initialize phospho
phospho.init(
    api_key=os.getenv("PHOSPHO_API_KEY"), 
    project_id=os.getenv("PHOSPHO_PROJECT_ID")
)

# Load data and initialize RAG once at startup
with open('exercise_data.json', 'r') as f:
    data = json.load(f)
rag = ExerciseRAG(data)

# Security
bearer = HTTPBearer()

def get_api_key(authorization: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    api_key_token = authorization.credentials
    if api_key_token != os.getenv("MY_SECRET_KEY"):
        raise HTTPException(status_code=401, detail="Invalid token")

# Request/Response Models
class Message(BaseModel):
    message: str

class WorkoutPlanRequest(BaseModel):
    training_frequency: int
    available_equipment: Optional[List[str]] = None
    experience_level: Optional[str] = None
    intensity: Optional[str] = "moderate"
    target_body_parts: Optional[List[str]] = None
    session_duration: Optional[int] = 60

class SimilarExerciseRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5

class WorkoutPromptRequest(BaseModel):
    prompt: str
    available_equipment: Optional[List[str]] = None
    experience_level: Optional[str] = None
    intensity: Optional[str] = 'moderate'
    target_body_parts: Optional[List[str]] = None
    num_exercises: Optional[int] = 5

class FormattedWorkoutRequest(WorkoutPromptRequest):
    format_style: Optional[str] = "detailed" # or "concise"

# Endpoints
@app.get("/")
def read_root():
    return {"Hello": "Exercise RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/metadata")
def get_metadata():
    """Get available options for equipment, body parts, etc."""
    return {
        "body_parts": rag.body_parts,
        "equipment_types": rag.equipment_types,
        "difficulty_levels": rag.difficulty_levels,
        "exercise_types": rag.exercise_types
    }

@app.post("/workout-plan")
def generate_workout_plan(request: WorkoutPlanRequest):
    try:
        plan = rag.generate_workout_plan(
            training_frequency=request.training_frequency,
            available_equipment=request.available_equipment,
            experience_level=request.experience_level,
            intensity=request.intensity,
            target_body_parts=request.target_body_parts,
            session_duration=request.session_duration
        )
        
        # Log to analytics
        phospho.log(
            input=str(request.dict()),
            output=str(plan)
        )
        
        return JSONResponse(content=plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/similar-exercises")
def find_similar_exercises(request: SimilarExerciseRequest):
    try:
        similar = rag.find_similar_exercises(
            query=request.query,
            n=request.num_results
        )
        
        # Log to analytics
        phospho.log(
            input=request.query,
            output=str(similar)
        )
        
        return JSONResponse(content=similar)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend-workout")
def recommend_workout(request: WorkoutPromptRequest):
    try:
        exercises = rag.recommend_exercises(
            prompt=request.prompt,
            available_equipment=request.available_equipment,
            experience_level=request.experience_level,
            intensity=request.intensity,
            target_body_parts=request.target_body_parts,
            top_k=request.num_exercises
        )
        
        # Log to analytics
        phospho.log(
            input=str(request.dict()),
            output=str(exercises)
        )
        
        return exercises
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def format_workout_with_gpt(exercises: List[Dict[str, Any]], style: str) -> str:
    exercises_str = "\n".join([
        f"- {ex['Title']}: {ex['Desc']}" 
        for ex in exercises
    ])
    
    prompt = f"""Given these exercises:
    {exercises_str}
    
    Create a {style} workout format that includes:
    - Exercise name
    - Brief description
    - Sets and reps recommendations
    - Form tips
    - Rest periods
    
    Make it motivating and easy to follow.
    """
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable personal trainer helping create workout plans.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    # Extract the message from the completion
    answer = completion.choices[0].message.content

    # Log to analytics
    phospho.log(input=prompt, output=answer)

    return {"answer": answer}

def format_workout_with_groq(exercises: List[Dict[str, Any]], style: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    exercises_str = "\n".join([
        f"- {ex['Title']}: {ex['Desc']}" 
        for ex in exercises
    ])
    
    prompt = f"""Given these exercises:
    {exercises_str}
    
    Create a {style} workout format that includes:
    - Exercise name
    - Brief description
    - Sets and reps recommendations
    - Form tips
    - Rest periods
    
    Make it motivating and easy to follow.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",  # Using llama2 model
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable personal trainer helping create workout plans.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1024
    )

    answer = completion.choices[0].message.content

    # Log to analytics
    phospho.log(input=prompt, output=answer)

    return {"answer": answer, "exercises": exercises}

@app.post("/recommend-workout-formatted")
def recommend_workout_formatted(request: FormattedWorkoutRequest):
    try:
        exercises = rag.recommend_exercises(
            prompt=request.prompt,
            available_equipment=request.available_equipment,
            experience_level=request.experience_level,
            intensity=request.intensity,
            target_body_parts=request.target_body_parts,
            top_k=request.num_exercises
        )
        
        formatted_response = format_workout_with_groq(
            exercises.slice(0, request.num_exercises),
            request.format_style
        )
        
        return formatted_response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)