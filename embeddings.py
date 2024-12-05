import openai
import numpy as np

def generate_prompt_embedding(prompt):
    """Generate embedding for the given prompt."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=prompt
    )
    return response['data'][0]['embedding']

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def select_relevant_insights(prompt, insights_embeddings):
    """
    Select relevant insights based on embedding similarity.
    
    Args:
        prompt (str): The input prompt text.
        insights_embeddings (list of dict): List of dictionaries containing insight ID, text, and precomputed embedding.
            Example:
            [
                {"id": 1, "text": "Insight 1 text", "embedding": [0.1, 0.2, ...]},
                {"id": 2, "text": "Insight 2 text", "embedding": [0.3, 0.4, ...]},
            ]
    
    Returns:
        list of dict: Top N relevant insights with their IDs, text, and similarity scores.
    """
    # Generate embedding for the prompt
    prompt_embedding = generate_prompt_embedding(prompt)
    
    insights_with_scores = []
    for insight in insights_embeddings:
        similarity = cosine_similarity(prompt_embedding, insight['embedding'])
        insights_with_scores.append({
            "id": insight['id'],
            "text": insight['text'],
            "similarity": similarity
        })

    # Sort by similarity and return the top 2 insights
    sorted_insights = sorted(insights_with_scores, key=lambda x: x['similarity'], reverse=True)
    return sorted_insights[:2]  # Adjust the number of insights as needed
