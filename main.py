from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import PyPDF2
from google import genai
from google.genai import types
import json,os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)

# ⚠️ NEVER expose API keys in production
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_stream):
    text = ""
    reader = PyPDF2.PdfReader(file_stream)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()
@app.route("/")
def home():
    return render_template("index.html")
# ---------------- ANALYZE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400

    resume_file = request.files["resume"]
    jd = request.form.get("job_description", "")

    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text:
        return jsonify({"error": "Could not read resume"}), 400

    prompt = f"""
You are an expert ATS scanner.

Return ONLY valid JSON in EXACTLY this format:

{{
  "score": number,
  "skills": [string],
  "analysis": [
    {{
      "title": string,
      "status": "pass" | "warn" | "fail",
      "desc": string
    }}
  ]
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd if jd else "N/A"}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        data = json.loads(response.text)

        # SAFETY NET (never crash frontend)
        return jsonify({
            "score": data.get("score", 0),
            "skills": data.get("skills", []),
            "analysis": data.get("analysis", [])
        })

    except Exception as e:
        return jsonify({
            "score": 0,
            "skills": [],
            "analysis": [{
                "title": "Analysis Error",
                "status": "fail",
                "desc": str(e)
            }]
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
