from flask import Flask, jsonify, request, Response
from embeddings import generate_prompt_embedding, select_relevant_insights
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
    if not prompt or not insights:
        return jsonify({"error": "Missing 'prompt' or 'insights'"}), 400

    try:
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model, api_key=OPENAI_API_KEY)
        
        # Generate embeddings for the insights (you should have precomputed them and sent them)
        insights_embeddings = [{"id": insight['id'], "text": insight['text'], "embedding": insight['embedding']} for insight in insights]
        
        # Select relevant insights
        relevant_insights = select_relevant_insights(prompt, insights_embeddings)

        # Combine relevant insights and prompt
        relevant_text = "\n".join([insight['text'] for insight in relevant_insights])
        combined_input = f"Relevant Insights:\n{relevant_text}\n\nPrompt:\n{prompt}"

        # Define your template
        template = """
Analyze the content of the provided question and generate insights. Include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format without a 'json' heading, with each insight structured as follows and based on the provided input:

- Insight1:
  - Summary: Insight summary here
  - Description: Detailed insight description here
- Insight2:
  - Summary: Insight summary here
  - Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided question.
2. Do not introduce new elements or information not present in the question.
3. If there is no insight, generate the response without JSON header with the message: "Message": "There is no insight found. Please send a different question."
4. Ensure the response does not mention ChatGPT or OpenAI.
5. All insights should be within an array of objects labeled Insights. Inside this single object, each insight should be labeled as Insight1, Insight2, etc., with its respective Summary and Description fields.
6. If a specific number is mentioned in the question, that exact number of insights should be generated; otherwise, generate insights as appropriate.
{input}
        """

        # Construct prompt using the template and the question
        prompt = template.format(input=combined_input)

        # Send the prompt to the model
        response = llm(prompt)
        logging.debug(f"Raw model response: {response}")

        # Parse the response question as JSON
        try:
            response_json = json.loads(response.content)
            
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
