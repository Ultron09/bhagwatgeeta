import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, Response
from gtts import gTTS
import os
import json
import cv2
import numpy as np
import dlib
import mediapipe as mp
import time
from datetime import datetime
import threading
from deepface import DeepFace
import pdfkit
import uuid

# -----------------------------------------------------
# 1) Configure Gemini
# -----------------------------------------------------
genai.configure(api_key=os.getenv("GwminiKey")  )

instruction = (
    "Respond like a psychologist who integrates wisdom from the Bhagavad Gita into practical solutions. "
    "Ensure responses feel deeply human, empathetic, and emotionally resonant. Half of the response should "
    "offer a Geeta-integrated response while the other half (or less) should ask a thoughtful question to "
    "understand the user’s situation better and keep the conversation flowing naturally. Don't actually "
    "make it into 2 parts; put it up in a very smart manner without making it obvious. and yes return a corresponding verse which it related to their problem"
)

def create_chat():
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[
        {"role": "user", "parts": [instruction]}
    ])
    return chat

chat = create_chat()

app = Flask(__name__)

# -----------------------------------------------------
# 2) Global Psychological Report
# -----------------------------------------------------
psychological_report = {
    "analysis": "",
    "final_solution": "",
    "face_analysis": "",
    "integrated_report": ""
}

# -----------------------------------------------------
# 3) Load Bhagavad Gita Verses
# -----------------------------------------------------
try:
    with open("geeta_verses.json", "r", encoding="utf-8") as f:
        bhagavad_geeta_data = json.load(f)
except Exception as e:
    print("Error loading geeta verses file:", e)
    bhagavad_geeta_data = []

def get_bhagavad_geeta_verse(chapter, verse):
    for item in bhagavad_geeta_data:
        if item.get("chapter_number") == chapter and item.get("verse_number") == verse:
            return item
    return None

# -----------------------------------------------------
# 4) Flask Routes
# -----------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    """
    Chat route that:
      1) Takes user message
      2) Uses Gemini to respond
      3) Generates TTS for that response in a unique file
      4) Returns JSON with 'reply' and 'audio_url'
    """
    global chat, psychological_report
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # -- Chat with Gemini --
        response = chat.send_message(user_message)
        response_text = response.text
        print("[/chat] Gemini reply:", response_text)

        # -- Generate TTS for chat response in a unique file --
        audio_filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join("static", audio_filename)
        try:
            tts = gTTS(text=response_text, lang='en', slow=False, tld='co.in')
            tts.save(audio_path)
            print("[/chat] TTS generated at:", audio_path)
        except Exception as e_tts:
            print("[/chat] TTS generation failed:", e_tts)
            audio_path = ""  # fallback to empty if TTS fails

        # -- Additional prompts for analysis and final solution --
        analysis_prompt = (
            "Summarize the current psychological state of the user based on the conversation so far. "
            "Keep it brief yet insightful."
        )
        solution_prompt = (
            "Provide a final solution or advice based on the user's psychological state and the Bhagavad Gita's wisdom."
        )
        try:
            analysis_response = chat.send_message(analysis_prompt).text
            solution_response = chat.send_message(solution_prompt).text
            psychological_report["analysis"] = analysis_response
            psychological_report["final_solution"] = solution_response
        except Exception as e_analysis:
            print("[/chat] Analysis/solution prompts failed:", e_analysis)

        return jsonify({
            "reply": response_text,
            "audio_url": audio_path  # Always return audio_path, even if empty
        })

    except Exception as e:
        print("[/chat] Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/update_instruction', methods=['POST'])
def update_instruction():
    global instruction, chat
    try:
        instruction = request.json.get("instruction", "")
        chat = create_chat()
        return jsonify({"message": "Instruction updated successfully!"})
    except Exception as e:
        print("[/update_instruction] Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/psychological_report', methods=['GET'])
def get_psychological_report():
    """
    Integrates chat analysis, face analysis, and final solution
    into a single refined text stored in 'integrated_report'.
    """
    global psychological_report
    integrated_text = (
        "Chat Analysis:\n" + psychological_report.get("analysis", "") + "\n\n" +
        "Face Analysis:\n" + psychological_report.get("face_analysis", "") + "\n\n" +
        "Final Solution:\n" + psychological_report.get("final_solution", "")
    )
    try:
        refined = chat.send_message(
            "Please combine the following analyses into one concise final report without any questions or extra commentary:\n"
            + integrated_text
        ).text
        psychological_report["integrated_report"] = refined
    except Exception as e:
        print("[/psychological_report] Error refining integrated report:", e)
        refined = integrated_text
        psychological_report["integrated_report"] = refined

    return jsonify({"integrated_report": refined})

@app.route('/verse_search', methods=['POST'])
def verse_search():
    """
    Accepts "chapter.verse" (e.g. "2.16"), returns TTS for verse & explanation.
    Since gTTS does not support Sanskrit ('sa'), fallback to Hindi ('hi').
    """
    global chat
    try:
        verse_input = request.json.get("verse", "").strip()
        if not verse_input:
            return jsonify({"error": "Verse code cannot be empty"}), 400

        parts = verse_input.split('.')
        if len(parts) != 2:
            return jsonify({"error": "Invalid verse code format. Use Chapter.Verse, e.g. 2.16"}), 400

        chapter = int(parts[0].strip())
        verse = int(parts[1].strip())

        verse_data = get_bhagavad_geeta_verse(chapter, verse)
        if not verse_data:
            return jsonify({"error": "Verse not found"}), 404

        # TTS for the verse text
        verse_text = verse_data.get("text", "")
        verse_audio_filename = f"verse_{chapter}_{verse}_{uuid.uuid4().hex[:8]}.mp3"
        verse_audio_path = os.path.join("static", verse_audio_filename)
        try:
            # Attempt Sanskrit
            tts_obj = gTTS(text=verse_text, lang='sa', slow=False)
            tts_obj.save(verse_audio_path)
        except Exception as e_verse_tts:
            print("Verse TTS generation failed with 'sa':", e_verse_tts)
            # Fallback to Hindi
            tts_obj = gTTS(text=verse_text, lang='hi', slow=False)
            tts_obj.save(verse_audio_path)
        print("Verse TTS saved to:", verse_audio_path)

        # Explanation with Gemini
        explanation_prompt = (
            f"Please provide a concise explanation of the following Bhagavad Gita verse. "
            f"Do not include any questions or extra commentary. Just a direct explanation.\n\nVerse:\n{verse_text}"
        )
        gemini_explanation = ""
        try:
            gemini_explanation = chat.send_message(explanation_prompt).text
        except Exception as e_expl:
            print("Gemini explanation failed:", e_expl)
            gemini_explanation = "No explanation available."

        # TTS for explanation (English)
        explanation_audio_filename = f"explanation_{chapter}_{verse}_{uuid.uuid4().hex[:8]}.mp3"
        explanation_audio_path = os.path.join("static", explanation_audio_filename)
        try:
            tts_exp = gTTS(text=gemini_explanation, lang='en', slow=False, tld='co.in')
            tts_exp.save(explanation_audio_path)
        except Exception as e_exp_tts:
            print("Explanation TTS generation failed:", e_exp_tts)
            # fallback to basic English
            tts_exp = gTTS(text=gemini_explanation, lang='en', slow=False)
            tts_exp.save(explanation_audio_path)
        print("Explanation TTS saved to:", explanation_audio_path)

        return jsonify({
            "sanskrit": verse_text,
            "explanation": gemini_explanation,
            "audio_url": verse_audio_path,
            "explanation_audio_url": explanation_audio_path
        })

    except Exception as e:
        print("Error in /verse_search route:", e)
        return jsonify({"error": f"Verse search failed: {str(e)}"}), 500

# -----------------------------------------------------
# 5) Video Feed + Face Analysis
# -----------------------------------------------------
video_capture = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def gen_frames():
    start_time = time.time()
    emotion_history = []
    posture_history = []
    gesture_history = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Emotions
        dominant_emotion = None
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis:
                dominant_emotion = analysis[0]['dominant_emotion']
        except:
            dominant_emotion = None
        if dominant_emotion:
            emotion_history.append(dominant_emotion)

        # Posture
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
            nose = landmarks[0]
            head_position = nose.y
            posture = {
                'shoulder_alignment': shoulder_alignment,
                'head_position': head_position,
                'is_straight': shoulder_alignment < 0.05
            }
            posture_history.append(posture)

        # Gestures
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            gestures = []
            if abs(left_wrist.x - right_wrist.x) < 0.1:
                gestures.append("crossed_arms")
            nose = landmarks[0]
            left_hand = landmarks[19]
            right_hand = landmarks[20]
            if (abs(left_hand.x - nose.x) < 0.1 or 
                abs(right_hand.x - nose.x) < 0.1):
                gestures.append("touching_face")
            if gestures:
                gesture_history.extend(gestures)

        # Every 30s refine face analysis
        if time.time() - start_time >= 30:
            deductions = ""
            if emotion_history:
                dominant = max(set(emotion_history), key=emotion_history.count)
                deductions += f"Dominant emotion: {dominant}. "
            if posture_history:
                straight_count = sum(1 for p in posture_history if p.get('is_straight'))
                ratio = straight_count / len(posture_history)
                deductions += f"Upright posture: {ratio*100:.1f}% of time. "
            if gesture_history:
                unique_gestures = set(gesture_history)
                deductions += "Gestures: " + ", ".join(unique_gestures) + "."

            prompt = (
                "Refine the following face analysis into a concise summary "
                "without any questions:\n" + deductions
            )
            try:
                refined = chat.send_message(prompt).text
            except:
                refined = deductions
            psychological_report["face_analysis"] = refined

            start_time = time.time()
            emotion_history.clear()
            posture_history.clear()
            gesture_history.clear()

        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------
# 6) Download PDF
# -----------------------------------------------------
@app.route('/download_report', methods=['GET'])
def download_report():
    """
    Creates a formal PDF with labeled sections for:
      - Chat Analysis
      - Face Analysis
      - Final Solution
      - Integrated Report
    """
    chat_analysis = psychological_report.get("analysis", "No chat analysis.")
    face_analysis = psychological_report.get("face_analysis", "No face analysis.")
    final_solution = psychological_report.get("final_solution", "No final solution.")
    integrated_report = psychological_report.get("integrated_report", "No integrated report.")

    chat_analysis_html = chat_analysis.replace("\n", "<br>")
    face_analysis_html = face_analysis.replace("\n", "<br>")
    final_solution_html = final_solution.replace("\n", "<br>")
    integrated_report_html = integrated_report.replace("\n", "<br>")

    html = f"""<html>
<head>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      color: #333;
    }}
    h1 {{
      text-align: center;
      color: #333;
      margin-bottom: 0;
    }}
    .timestamp {{
      font-size: 12px;
      color: #666;
      text-align: center;
      margin-top: 0;
      margin-bottom: 20px;
    }}
    .section {{
      margin-bottom: 20px;
    }}
    .section h2 {{
      margin: 10px 0 5px 0;
      color: #222;
      font-size: 18px;
    }}
    .section p {{
      font-size: 14px;
      line-height: 1.5;
      text-align: justify;
    }}
    hr {{
      margin: 20px 0;
      border: none;
      border-top: 1px solid #aaa;
    }}
    .footer {{
      margin-top: 30px;
      text-align: center;
      font-size: 12px;
      color: #777;
    }}
  </style>
</head>
<body>
  <h1>Behavioral Analysis Report</h1>
  <p class="timestamp">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <hr/>

  <div class="section">
    <h2>Chat Analysis</h2>
    <p>{chat_analysis_html}</p>
  </div>

  <div class="section">
    <h2>Face Analysis</h2>
    <p>{face_analysis_html}</p>
  </div>

  <div class="section">
    <h2>Final Solution</h2>
    <p>{final_solution_html}</p>
  </div>

  <div class="section">
    <h2>Integrated Report</h2>
    <p>{integrated_report_html}</p>
  </div>

  <hr/>
  <div class="footer">
    <p>Generated by AI Behavioral Analysis System</p>
  </div>
</body>
</html>"""

    pdf = pdfkit.from_string(html, False)
    response = Response(pdf, mimetype="application/pdf")
    response.headers["Content-Disposition"] = "attachment; filename=Behavioral_Analysis_Report.pdf"
    return response

if __name__ == '__main__':
    app.run(debug=True)
