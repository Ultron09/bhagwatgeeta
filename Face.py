import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import dlib
from deepface import DeepFace
import mediapipe as mp
import time
from datetime import datetime
import threading

class BehaviorAnalyzer:
    def __init__(self, analysis_duration=30):  # Default 30 seconds
        self.root = Tk()
        self.root.title("Behavioral Analysis System")
        self.root.geometry("1200x800")
        
        # Analysis duration
        self.analysis_duration = analysis_duration
        self.start_time = None
        
        # Initialize detectors
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data collection
        self.emotion_history = []
        self.posture_history = []
        self.gesture_history = []
        self.micro_expressions = []
        self.confidence_indicators = []
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Video frame
        self.video_label = Label(self.root)
        self.video_label.pack(side=LEFT, padx=10, pady=10)
        
        # Timer label
        self.timer_label = Label(self.root, 
                               text=f"Analysis starts in: {self.analysis_duration}s",
                               font=('Arial', 14))
        self.timer_label.pack(pady=5)
        
        # Status label
        self.status_label = Label(self.root, 
                                text="Preparing analysis...",
                                font=('Arial', 12))
        self.status_label.pack(pady=5)

    def analyze_emotions(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if analysis:
                return analysis[0]['dominant_emotion'], analysis[0]['emotion']
            return None, None
        except:
            return None, None

    def analyze_posture(self, results):
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Analyze shoulder alignment
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
            
            # Analyze head position
            nose = landmarks[0]
            head_position = nose.y
            
            return {
                'shoulder_alignment': shoulder_alignment,
                'head_position': head_position,
                'is_straight': shoulder_alignment < 0.05
            }
        return None

    def analyze_gestures(self, results):
        gestures = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check for crossed arms
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            if abs(left_wrist.x - right_wrist.x) < 0.1:
                gestures.append("crossed_arms")
            
            # Check for hand touching face
            nose = landmarks[0]
            left_hand = landmarks[19]
            right_hand = landmarks[20]
            if (abs(left_hand.x - nose.x) < 0.1 or 
                abs(right_hand.x - nose.x) < 0.1):
                gestures.append("touching_face")
        
        return gestures

    def make_deductions(self):
        detailed_analysis = []
        
        # Analyze emotional patterns with detailed explanations
        if self.emotion_history:
            dominant_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            emotion_variety = len(set(self.emotion_history))
            emotion_changes = sum(1 for i in range(1, len(self.emotion_history)) 
                                if self.emotion_history[i] != self.emotion_history[i-1])
            
            # Emotional Pattern Analysis
            emotional_analysis = "Emotional Pattern Analysis:\n"
            if emotion_variety <= 2:
                emotional_analysis += "- Subject displays limited emotional range, showing only " \
                                    f"{emotion_variety} distinct emotions. This emotional restraint " \
                                    "could indicate either strong emotional control or potential " \
                                    "suppression of genuine feelings, common in individuals who " \
                                    "prioritize maintaining a composed facade.\n"
            elif emotion_changes > len(self.emotion_history) * 0.3:
                emotional_analysis += "- Notable emotional volatility observed, with frequent shifts " \
                                    "between different emotional states. This pattern often suggests " \
                                    "heightened emotional sensitivity or difficulty in emotional regulation, " \
                                    "possibly due to current situational stressors or underlying anxiety.\n"
            
            if dominant_emotion == 'neutral':
                emotional_analysis += "- Predominant neutral expression might indicate:\n" \
                                    "  * Professional training or experience in maintaining composure\n" \
                                    "  * Possible emotional guardedness or defensive mechanism\n" \
                                    "  * Cultural background emphasizing emotional restraint\n"
            detailed_analysis.append(emotional_analysis)
            
            # Micro-Expression Analysis
            micro_expr_analysis = "Micro-Expression Analysis:\n"
            fleeting_emotions = [e for e in self.emotion_history if self.emotion_history.count(e) <= 2]
            if fleeting_emotions:
                micro_expr_analysis += f"- Detected brief displays of {', '.join(set(fleeting_emotions))}. " \
                                     "These momentary expressions often reveal underlying emotions that " \
                                     "the subject may be attempting to conceal.\n"
                if 'surprise' in fleeting_emotions:
                    micro_expr_analysis += "- Fleeting surprise expressions might indicate:\n" \
                                         "  * Unexpected information processing\n" \
                                         "  * Cognitive dissonance between expected and actual events\n" \
                                         "  * Momentary breaks in composed facade\n"
            detailed_analysis.append(micro_expr_analysis)
            
            # Body Language Deep Analysis
            body_language = "Comprehensive Body Language Analysis:\n"
            
            # Analyze gestures
            gesture_patterns = {}
            for gesture in self.gesture_history:
                gesture_patterns[gesture] = gesture_patterns.get(gesture, 0) + 1
            
            if 'touching_face' in gesture_patterns:
                touch_frequency = gesture_patterns['touching_face']
                body_language += f"- Face touching observed {touch_frequency} times:\n" \
                               "  * This self-soothing behavior often indicates internal discomfort\n" \
                               "  * May be attempting to create a physical barrier when feeling exposed\n" \
                               "  * Could be a habitual stress response pattern\n"
            
            if 'crossed_arms' in gesture_patterns:
                cross_frequency = gesture_patterns['crossed_arms']
                body_language += f"- Arms crossed {cross_frequency} times:\n" \
                               "  * Classic defensive posture indicating psychological barriers\n" \
                               "  * Might be seeking emotional security or comfort\n" \
                               "  * Could indicate disagreement or skepticism\n"
            
            # Analyze posture changes
            if self.posture_history:
                straight_posture = sum(1 for p in self.posture_history if p.get('is_straight', False))
                posture_ratio = straight_posture / len(self.posture_history)
                
                if posture_ratio > 0.8:
                    body_language += "- Consistently upright posture (>80% of time):\n" \
                                   "  * Indicates high self-confidence and assertiveness\n" \
                                   "  * Suggests professional training or military background\n" \
                                   "  * May be consciously maintaining formal bearing\n"
                elif posture_ratio < 0.4:
                    body_language += "- Frequently relaxed or slouched posture:\n" \
                                   "  * Suggests comfort in the environment\n" \
                                   "  * May indicate fatigue or low energy levels\n" \
                                   "  * Could be a sign of casual or informal attitude\n"
            detailed_analysis.append(body_language)
            
            # Confidence and Power Dynamics
            power_dynamics = "Confidence and Power Dynamics Analysis:\n"
            
            # Analyze eye contact patterns
            if self.confidence_indicators:
                eye_contact_score = sum(1 for c in self.confidence_indicators if c > 0.7)
                eye_contact_ratio = eye_contact_score / len(self.confidence_indicators)
                
                if eye_contact_ratio > 0.7:
                    power_dynamics += "- Strong, consistent eye contact (>70% of interaction):\n" \
                                    "  * Projects confidence and authority\n" \
                                    "  * Indicates comfort with interpersonal engagement\n" \
                                    "  * Suggests leadership qualities or experience\n"
                elif eye_contact_ratio < 0.3:
                    power_dynamics += "- Limited eye contact observed:\n" \
                                    "  * May indicate submission or deference\n" \
                                    "  * Could suggest social anxiety or discomfort\n" \
                                    "  * Possible cultural influence on eye contact norms\n"
            
            # Analyze spatial behavior
            if self.posture_history:
                leaning_forward = sum(1 for p in self.posture_history if p.get('head_position', 0) < 0.4)
                if leaning_forward > len(self.posture_history) * 0.3:
                    power_dynamics += "- Frequent forward leaning:\n" \
                                    "  * Shows engagement and interest\n" \
                                    "  * Indicates desire to connect or persuade\n" \
                                    "  * May suggest dominance behavior\n"
            detailed_analysis.append(power_dynamics)
            
            # Overall Behavioral Profile
            profile = "\nOverall Behavioral Profile:\n"
            profile += "This individual appears to be "
            
            # Compile behavioral traits
            traits = []
            if emotion_variety <= 2:
                traits.append("emotionally guarded")
            if 'crossed_arms' in gesture_patterns:
                traits.append("defensive")
            if eye_contact_ratio > 0.7 if self.confidence_indicators else False:
                traits.append("confident")
            if gesture_patterns.get('touching_face', 0) > 3:
                traits.append("potentially anxious")
                
            profile += ", ".join(traits) if traits else "displaying mixed behavioral signals"
            profile += ".\n\nKey Behavioral Indicators:\n"
            
            # Add specific behavioral indicators
            if traits:
                for trait in traits:
                    if trait == "emotionally guarded":
                        profile += "- Limited emotional expression suggests careful control over self-presentation\n"
                    elif trait == "defensive":
                        profile += "- Physical barriers through posture indicate psychological guardedness\n"
                    elif trait == "confident":
                        profile += "- Strong eye contact and posture suggest leadership qualities\n"
                    elif trait == "potentially anxious":
                        profile += "- Self-soothing behaviors point to underlying tension or discomfort\n"
                    
            detailed_analysis.append(profile)
        
        return "\n".join(detailed_analysis)

    def update_frame(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.analysis_duration - elapsed_time)
        
        ret, frame = self.cap.read()
        if ret:
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            # Analyze frame
            dominant_emotion, emotions = self.analyze_emotions(frame)
            if dominant_emotion:
                self.emotion_history.append(dominant_emotion)
            
            posture = self.analyze_posture(results)
            if posture:
                self.posture_history.append(posture)
            
            gestures = self.analyze_gestures(results)
            if gestures:
                self.gesture_history.extend(gestures)
            
            # Update UI
            self.timer_label.config(
                text=f"Time remaining: {int(remaining_time)}s")
            
            # Convert frame for display
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
            
            # Check if analysis is complete
            if remaining_time <= 0:
                self.complete_analysis()
                return
            
            self.root.after(10, self.update_frame)

    def complete_analysis(self):
        # Generate final deductions
        deductions = self.make_deductions()
        
        # Create detailed report
        report = f"""
Behavioral Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Detailed Observations and Deductions:
{deductions}

Analysis based on:
- Emotional patterns over {self.analysis_duration} seconds
- Posture and gesture analysis
- Micro-expression detection
- Confidence indicators
"""
        
        # Save report to file
        with open("behavioral_analysis_report.txt", "w") as f:
            f.write(report)
        
        print("\nAnalysis Complete!")
        print(report)
        
        # Cleanup and close
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.update_frame()
        self.root.mainloop()

if __name__ == "__main__":
    # Run analysis for 30 seconds
    analyzer = BehaviorAnalyzer(analysis_duration=30)
    analyzer.run() 