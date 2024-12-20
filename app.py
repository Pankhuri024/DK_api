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
    # Parse request data
    try:
        data = request.get_json()
        question = data.get("question")
        insights = data.get("insights")
        if not question or not insights:
            return jsonify({"message": "Missing 'question' or 'insights' in the request body."}), 400
    except Exception as e:
        return jsonify({"message": "Invalid request format. Ensure the JSON is structured correctly."}), 400
    # Prepare prompt for OpenAI
    formatted_prompt = generate_prompt(question, insights)
    try:
        # Initialize OpenAI model
        selected_model = 'gpt-3.5-turbo'  # Example model, adjust as needed
        llm = ChatOpenAI(model=selected_model, api_key=OPENAI_API_KEY)
        response = llm(formatted_prompt)
        # Access the generated content properly
        response_text = response.content.strip()  # This should be the correct attribute
        # Parse the response text as JSON
        try:
            response_json = json.loads(response_text)
            new_insights = response_json.get('Insights', [])
            if not new_insights:
                return jsonify({"message": "There is no insight found. Please send a different prompt."}), 200
            return jsonify(new_insights), 200
        except json.JSONDecodeError:
            return jsonify({"message": "Error parsing response as JSON."}), 500
    except Exception as e:
        if "insufficient_quota" in str(e):
            return jsonify({"message": "Quota exceeded. Please check your OpenAI plan and billing details."}), 429
        return jsonify({"message": "Error processing the prompt."}), 500


def generate_prompt(question, insights):
    """
    Generates a structured prompt for OpenAI based on the user question and provided insights.
    """
    prompt = f"""
    You are an AI that analyzes user questions and existing insights to generate new insights. 
    Based on the following user question and insights, generate new insights in JSON format with the fields 'ID', 'Summary', 'Description', and 'Source_Insights'. The response should follow the specified syntax.

    Question: "{question}"

    Existing Insights:
    {json.dumps(insights, indent=2)}

    - Follow this syntax for the generated insights:
      "Insights":[
        {{
          "Summary": "<Summary of the insight>",
          "Description": "<Detailed description of the insight>",
          "Source_Insights": <List of IDs from the existing insights that are relevant to this new insight>
        }},
        ...
      ]
    - Analyze the content of the provided question and insights.
    - Use the insights to create new insights relevant to the question.
    - Each new insight must include:
      - ID: The ID of the original insight.
      - Summary: A short summary (max 200 characters).
      - Description: A detailed description (max 1500 characters).
      - Source_Insights: A list of IDs from the existing insights that are used to derive the new insight (can be empty if not applicable).
    - If no relevant insights can be generated, respond with: {{"message": "There is no insight found. Please send a different prompt."}}.
    - If a specific number is mentioned in the question, that exact number of insights should be generated; otherwise, generate insights as appropriate.
    """
    return prompt




if __name__ == "__main__":
    app.run(debug=True)
