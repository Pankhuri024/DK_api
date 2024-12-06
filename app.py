from flask import Flask, jsonify, request, Response
import logging
import os
import openai
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable is not set.")
    raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

openai.api_key = OPENAI_API_KEY


@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    """
    Endpoint to generate new insights based on a user-provided question and existing insights.
    """
    logging.debug("Starting generate_insights function.")

    # Parse request data
    try:
        data = request.get_json()
        question = data.get("question")
        insights = data.get("insights")
        
        if not question or not insights:
            logging.error("Missing required fields: 'question' and 'insights'.")
            return jsonify({"message": "Missing 'question' or 'insights' in the request body."}), 400

        logging.debug(f"Received question: {question}")
        logging.debug(f"Received insights: {insights}")

    except Exception as e:
        logging.error(f"Error parsing request data: {e}")
        return jsonify({"message": "Invalid request format. Ensure the JSON is structured correctly."}), 400

    # Prepare prompt for OpenAI
    formatted_prompt = generate_prompt(question, insights)
    logging.debug(f"Formatted prompt: {formatted_prompt}")

    try:
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model, api_key=OPENAI_API_KEY)
        response = llm(formatted_prompt)
        logging.debug(f"Raw model response: {response}")

        # Access the generated content properly
        response_text = response.content.strip()  # This should be the correct attribute
        logging.debug(f"Generated text: {response_text}")

        # Parse the response text as JSON
        try:
            response_json = json.loads(response_text)
            # Handle different structures of "Insights"
            if "Insights" in response_json:
                insights = response_json["Insights"]

                if isinstance(insights, dict):
                    # Check for nested "Insights" or "insights"
                    if "Insights" in insights:
                        # Flatten nested "Insights"
                        response_json["Insights"] = insights["Insights"]
                    elif "insights" in insights:
                        # Flatten nested "insights" (lowercase)
                        response_json["Insights"] = insights["insights"]

            # Return the response_json directly
            return jsonify({"Insights": response_json}), 200

        except json.JSONDecodeError:
            logging.error("Error parsing generated text as JSON.")
            return jsonify({"message": "Error parsing response as JSON."}), 500

    except Exception as e:
        logging.error(f"Error processing the prompt: {e}")
        if "insufficient_quota" in str(e):
            return jsonify({"message": "Quota exceeded. Please check your OpenAI plan and billing details."}), 429
        return jsonify({"message": "Error processing the prompt."}), 500


def generate_prompt(question, insights):
    """
    Generates a structured prompt for OpenAI based on the user question and provided insights.
    """
    prompt = f"""
     You are an AI that analyzes user questions and existing insights to generate new insights. 
    Based on the following user question and insights, generate new insights in JSON format with the fields 'ID', 'Summary', 'Description', 'Source_Insights', and 'Relation_To_Question'.

    Question: "{question}"

    Existing Insights:
    {json.dumps(insights, indent=2)}

    Instructions:
    - Analyze the content of the provided question and insights.
    - Use the insights to create new insights relevant to the question.
    - Each new insight must include:
      - Summary: A short summary (max 200 characters).
      - Description: A detailed description (max 1500 characters).
      - Source_Insights: List of IDs of the existing insights that were used to generate this new insight.
    - If the insights are related to multiple existing insights, mention how they work together to answer the user's question.
    - The insights should be structured in such a way that they are answering the user's query directly and clearly. 
    - If no relevant insights can be generated, you should inform the user that no new insights were found.
    """
    return prompt



if __name__ == "__main__":
    app.run(debug=True)
