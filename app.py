from flask import Flask, jsonify, request, Response
# from embeddings import generate_prompt_embedding, select_relevant_insights
import openai
import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    logging.debug("Starting generate_insights function.")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        return jsonify({'message': 'OPENAI_API_KEY environment variable is not set.'}), 500
    
    data = request.get_json()
    # Validate input
    prompt = data.get("prompt")
    insights = data.get("insights")
    if not prompt or not isinstance(insights, list):
        return jsonify({"error": "Invalid input: 'prompt' must be a string and 'insights' must be a list"}), 400


    try:
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model, api_key=OPENAI_API_KEY)
        
        # Create a formatted input based on 'summary' and 'description'
        insights_text = "\n".join(
            f"- ID: {insight['id']}\n  Summary: {insight['summary']}\n  Description: {insight['description']}"
            for insight in insights
        )
        logging.debug("Starting insights_text ",insights_text)
        combined_input = f"Relevant Insights:\n{insights_text}\n\nPrompt:\n{prompt}"

        # Define your prompt template
        template = """
Analyze the content of the provided question and insights, and generate new insights based on them.
Each insight should include:
- ID: The ID of the original insight
- Summary: A short summary of the new insight (max 200 characters)
- Description: A detailed description of the new insight (max 1500 characters)

Instructions:
1. Use the provided question and insights to generate insights.
2. If no relevant insights are found, respond with: {"message": "There is no insight found. Please send a different question."}
3. Format your response as a JSON object with an array of new insights labeled "Insights".

Provided Insights:
{insights}

Prompt:
{prompt}

Output:
"""

        # Format the template
        formatted_prompt = template.format(insights=insights_text, prompt=prompt)
        logging.debug(f"Formatted prompt: {formatted_prompt}")


        # Send the prompt to the OpenAI model
        response = llm(formatted_prompt)
        logging.debug(f"Raw model response: {response}")

        # Parse the response question as JSON
        try:
            response_json = json.loads(response)
            
            # Extract 'Insights' from the JSON response
            insights = response_json.get('Insights', [])
            if not insights:
                # If no insights are found, return a message
                insights_json = json.dumps({"message": "There is no insight found. Please send a different question."})
            else:
                # Return the insights along with the IDs as a JSON response
                insights_json = json.dumps({"Insights": insights}, indent=2)

        except json.JSONDecodeError:
            insights_json = json.dumps({"message": "Error parsing response as JSON."})

        # Return the insights as a JSON response
        return Response(insights_json, mimetype='application/json')

    except Exception as e:
        if "insufficient_quota" in str(e):
            logging.error("Quota exceeded: Please check your OpenAI plan and billing details.")
            return jsonify({'message': 'Quota exceeded. Please check your OpenAI plan and billing details.'}), 429
        logging.error(f"Error processing question: {e}")
        return jsonify({'message': 'Error processing question'}), 500

if __name__ == "__main__":
    app.run(debug=True)
