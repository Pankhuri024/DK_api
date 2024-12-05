from flask import Flask, request, jsonify
from embeddings import generate_prompt_embedding, select_relevant_insights
import openai
import os

app = Flask(__name__)

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    data = request.get_json()

    # Validate input
    prompt = data.get("prompt")
    insights = data.get("insights")
    if not prompt or not insights:
        return jsonify({"error": "Missing 'prompt' or 'insights'"}), 400

    try:
        # Select relevant insights
        relevant_insights = select_relevant_insights(prompt, insights)

        # Combine relevant insights and prompt
        relevant_text = "\n".join([insight['text'] for insight in relevant_insights])
        combined_input = f"Relevant Insights:\n{relevant_text}\n\nPrompt:\n{prompt}"

        # Generate new insights using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant generating insights."},
                {"role": "user", "content": combined_input}
            ],
            max_tokens=500,
            temperature=0.7
        )

        generated_insight = response['choices'][0]['message']['content'].strip()

        return jsonify({
            "generated_insight": generated_insight,
            "used_insights": [{"id": insight['id'], "text": insight['text']} for insight in relevant_insights]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
