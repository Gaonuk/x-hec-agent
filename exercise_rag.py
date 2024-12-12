import os
from pinecone import Pinecone, ServerlessSpec  # Add ServerlessSpec import
from dotenv import load_dotenv
from typing import List, Dict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

class ExerciseRAG:
    def __init__(self, exercises_data: List[Dict]):
        load_dotenv()
        
        # Initialize Pinecone with new pattern
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        self.index_name = "exercises"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=2223,  # Adjust based on your vectorizer output
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
            
        self.index = self.pc.Index(self.index_name)
        self.exercises = exercises_data
        self.vectorizer = TfidfVectorizer()
        self._create_vectors()
        
    def initialize_vectors(self):
        # Create exercise descriptions
        descriptions = [f"{ex['Title']} {ex['Desc']} {ex['BodyPart']} {ex['Equipment']} {ex['Level']}" for ex in self.exercises]
        
        # Generate vectors
        vectors = self.vectorizer.fit_transform(descriptions).toarray()
        
        # Upload vectors to Pinecone
        batch_size = 100
        for i in range(0, len(self.exercises), batch_size):
            batch = list(zip(
                [str(j) for j in range(i, min(i + batch_size, len(self.exercises)))],
                vectors[i:i + batch_size].tolist()
            ))
            self.index.upsert(vectors=batch)

    def recommend_exercises(self, prompt: str, **params) -> List[Dict]:
        # Generate vector for prompt
        prompt_vector = self.vectorizer.transform([prompt]).toarray()[0]
        
        # Query Pinecone
        results = self.index.query(
            vector=prompt_vector.tolist(),
            top_k=50,
            include_metadata=True
        )
        
        # Filter results based on params
        filtered_exercises = []
        for match in results.matches:
            exercise = self.exercises[int(match.id)]
            if self._matches_params(exercise, **params):
                filtered_exercises.append((exercise, match.score))
                
        # Sort and return top results
        filtered_exercises.sort(key=lambda x: x[1], reverse=True)
        return [ex[0] for ex in filtered_exercises]

    def _matches_params(self, exercise: Dict, 
                        available_equipment: List[str] = None,
                        experience_level: str = None, 
                        target_body_parts: List[str] = None,
                        **kwargs) -> bool:
        """
        Check if exercise matches the provided parameters.
        """
        if available_equipment and exercise['Equipment'] not in available_equipment:
            return False
            
        if experience_level and exercise['Level'] != experience_level:
            return False
            
        if target_body_parts and exercise['BodyPart'] not in target_body_parts:
            return False
            
        return True

    def _extract_unique_values(self, field: str) -> List[str]:
        """Extract unique values for a given field from exercise data."""
        return sorted(list(set(ex[field] for ex in self.exercises)))

    def _create_vectors(self):
        """Create TF-IDF vectors from exercise descriptions."""
        texts = [f"{ex['Title']} {ex['Desc']} {ex['BodyPart']} {ex['Equipment']} {ex['Level']}" for ex in self.exercises]
        self.vectors = self.vectorizer.fit_transform(texts)

    def _create_exercise_index(self) -> Dict:
        """Create an index of exercises grouped by body part."""
        index = defaultdict(list)
        for exercise in self.exercises:
            index[exercise['BodyPart']].append(exercise)
        return index

    def generate_workout_plan(self,
                              training_frequency: int,
                              available_equipment: List[str] = None,
                              experience_level: str = None,
                              intensity: str = 'moderate',
                              target_body_parts: List[str] = None,
                              session_duration: int = 60) -> Dict:
        """
        Generate a workout plan based on provided parameters.
        Intensity can be 'light', 'moderate', or 'heavy'
        """
        if available_equipment is None:
            available_equipment = []
        if target_body_parts is None:
            target_body_parts = self.body_parts

        workout_plan = {
            'frequency': training_frequency,
            'intensity': intensity,
            'equipment': available_equipment,
            'workouts': {}
        }

        workout_split = self._create_workout_split(target_body_parts, training_frequency)
        for day, body_parts in enumerate(workout_split, 1):
            daily_exercises = []
            for body_part in body_parts:
                suitable_exercises = [
                    ex for ex in self.exercise_index[body_part]
                    if ex['Equipment'] in available_equipment
                    and (not experience_level or ex['Level'] == experience_level)
                ]
                
                suitable_exercises.sort(key=lambda x: x['Rating'], reverse=True)
                num_exercises = min(3, len(suitable_exercises))
                if num_exercises > 0:
                    selected = random.sample(suitable_exercises[:6], num_exercises)
                    for exercise in selected:
                        daily_exercises.append({
                            'name': exercise['Title'],
                            'body_part': exercise['BodyPart'],
                            'equipment': exercise['Equipment'],
                            'sets': random.randint(3, 5),
                            'reps': random.randint(8, 12),
                            'rest': 60,
                            'rating': exercise['Rating']
                        })
            workout_plan['workouts'][f'Day {day}'] = {
                'target_body_parts': body_parts,
                'exercises': daily_exercises
            }
        return workout_plan

    def _create_workout_split(self, body_parts: List[str], frequency: int) -> List[List[str]]:
        """Create a balanced workout split based on body parts and frequency."""
        complementary_groups = {
            'push': ['Chest', 'Shoulders', 'Triceps'],
            'pull': ['Back', 'Biceps'],
            'legs': ['Quadriceps', 'Hamstrings', 'Calves'],
            'core': ['Abdominals', 'Lower Back']
        }
        
        if frequency <= 3:
            splits = [[] for _ in range(frequency)]
            for i, part in enumerate(body_parts):
                splits[i % frequency].append(part)
        else:
            splits = []
            current_group = []
            for group in complementary_groups.values():
                matching_parts = [p for p in body_parts if p in group]
                if matching_parts:
                    if len(current_group) >= len(body_parts) // frequency:
                        splits.append(current_group)
                        current_group = []
                    current_group.extend(matching_parts)
            if current_group:
                splits.append(current_group)
            remaining = [p for p in body_parts if not any(p in g for g in complementary_groups.values())]
            for i, part in enumerate(remaining):
                if i < len(splits):
                    splits[i].append(part)
                else:
                    splits.append([part])
        return splits[:frequency]

    def find_similar_exercises(self, query: str, n: int = 5) -> List[Dict]:
        """Find exercises similar to the query."""
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = similarities.argsort()[-n:][::-1]
        
        return [
            {**self.exercises[idx], 'similarity': similarities[idx]}
            for idx in top_indices
        ]

# Example usage
if __name__ == "__main__":
    # Load data
    with open('exercise_data.json', 'r') as f:
        data = json.load(f)
    
    # Initialize system
    rag = ExerciseRAG(data)
    
    # Generate a workout plan
    plan = rag.generate_workout_plan(
        training_frequency=4,
        available_equipment=['Dumbbell', 'Barbell', 'Cable'],
        experience_level='Intermediate',
        intensity='moderate',
        target_body_parts=['Chest', 'Back', 'Legs', 'Shoulders'],
        session_duration=60
    )
    
    print(plan)
    
    # Print the plan
    print(f"\nWorkout Plan ({plan['frequency']} days per week)")
    print(f"Intensity: {plan['intensity']}")
    print(f"Equipment: {', '.join(plan['equipment'])}\n")
    
    for day, workout in plan['workouts'].items():
        print(f"\n{day} - {', '.join(workout['target_body_parts'])}")
        for exercise in workout['exercises']:
            print(f"- {exercise['name']}: {exercise['sets']} sets x {exercise['reps']} reps ({exercise['rest']}s rest)")