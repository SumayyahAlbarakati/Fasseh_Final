# prompt: full code give me flask code to take input video from upload.html page and annotate it using 4 Yolov8 models and display it in result.html

from ultralytics import YOLO
import cv2
from flask import Flask, render_template, request, Response, url_for
import json
import statistics
from matplotlib.patches import Wedge, Circle
import moviepy.editor as mp
import whisper

# Load the four YOLO models
model1 = YOLO('Confidence_Score.pt')
model2 = YOLO('Eyes_Gaze.pt')
model3 = YOLO('Face_Expression.pt')
model4 = YOLO('Stand.pt')


# Function to extract audio from the video
def extract_audio(video_path, audio_output_path):
  """Extracts audio from a video and saves it to a specified path.

  Args:
      video_path: Path to the video file.
      audio_output_path: Path to save the extracted audio file.
  """
  video = mp.VideoFileClip(video_path)
  video.audio.write_audiofile(audio_output_path)

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_path, model_type='large', language='en'):
  """Transcribes audio to text using the Whisper model.

  Args:
      audio_path: Path to the audio file.
      model_type: Whisper model type (e.g., 'base', 'medium', 'large').
      language: Language code for transcription (default: 'en').

  Returns:
      The transcribed text.
  """
  model = whisper.load_model(model_type)
  result = model.transcribe(audio_path, language=language)
  return result['text']

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_path, model_type='large', language='en'):
  """Transcribes audio to text using the Whisper model.

  Args:
      audio_path: Path to the audio file.
      model_type: Whisper model type (e.g., 'base', 'medium', 'large').
      language: Language code for transcription (default: 'en').

  Returns:
      The transcribed text.
  """
  model = whisper.load_model(model_type)
  result = model.transcribe(audio_path, language=language)
  return result['text']

# Main function to run extraction and transcription
def main(video_path, audio_output_path, transcription_output_path, model_type='large', language='ar'):
  """Main function to extract audio, transcribe it, and save the transcript.

  Args:
      video_path: Path to the video file.
      audio_output_path: Path to save the extracted audio file.
      transcription_output_path: Path to save the transcription.
      model_type: Whisper model type (e.g., 'base', 'medium', 'large').
      language: Language code for transcription (default: 'en').
  """
  print("Extracting audio from video...")
  extract_audio(video_path, audio_output_path)

  print("Transcribing audio to text...")
  transcription_text = transcribe_audio(audio_output_path, model_type, language)

  print("Writing transcription to file...")
  with open(transcription_output_path, 'w', encoding='utf-8') as file:
    file.write(transcription_text)

  print("Transcription complete. Check the transcription file for results.")


def calculate_average_confidences(data):
  """Calculates average confidence for each class in the given data.

  Args:
    data: A list of dictionaries, where each dictionary represents a model output and has the following keys:
      - "model": The name of the model.
      - "classes": A list of class names or a single class name.
      - "conf": A list of confidence scores or a single confidence score.

  Returns:
    A dictionary where keys are class names and values are average confidence scores.
  """

  class_confidences = {}

  for item in data:
    classes = item["classes"] if isinstance(item["classes"], list) else [item["classes"]]
    confs = item["conf"] if isinstance(item["conf"], list) else [item["conf"]]

    for i, cls in enumerate(classes):
      if cls not in class_confidences:
        class_confidences[cls] = []
      class_confidences[cls].append(confs[i])

  average_confidences = {cls: statistics.mean(confs) for cls, confs in class_confidences.items()}
  return average_confidences

import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display


def create_gauge_chart(data_dict, metric,title,color, filename):
  """Creates a gauge chart for the given metric and saves it as a separate file.

  Args:
    data_dict: A dictionary containing the data for different metrics.
    metric: The metric to create a gauge chart for.
    filename: The filename to save the chart as.
  """

  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_aspect('equal')
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.axis('off')

  # Gauge background
  wedge = Wedge((0.5, 0.5), 0.4, 0, 180, width=0.1, color='lightgray')
  ax.add_patch(wedge)

  # Gauge value
  value_angle = 180 * data_dict[metric]
  wedge_value = Wedge((0.5, 0.5), 0.4, 0, value_angle, width=0.1, color=color)  # Adjust color if needed
  ax.add_patch(wedge_value)

  # Needle
  needle_len = 0.35
  needle_x = 0.5 + needle_len * np.cos(np.deg2rad(value_angle))
  needle_y = 0.5 + needle_len * np.sin(np.deg2rad(value_angle))
  ax.plot([0.5, needle_x], [0.5, needle_y], color='black', linewidth=2)

  # Center circle
  circle = Circle((0.5, 0.5), radius=0.05, color='black')
  ax.add_patch(circle)

  # Text

  reshaped_text = arabic_reshaper.reshape(title)
  bidi_text = get_display(reshaped_text)
  xlbl = f"{bidi_text}\n\n\n\n\n\n"  # Adjust label as needed
  ax.text(0.5, 0.8, xlbl, ha='center', va='center', fontsize=24)
  ax.text(0.5, 0.3, f"{data_dict[metric]:.2f}", ha='center', va='center', fontsize=14)

  plt.savefig(f'static/{filename}')
  plt.close(fig)


# prompt: function to create a bar chart without eye stance and uncofident

def create_bar_chart(data_dict, filename):
    """Creates a bar chart for the given data and saves it as a separate file,
    excluding 'eye', 'stance', and 'unconfident' metrics.

    Args:
      data_dict: A dictionary containing the data for different metrics.
      filename: The filename to save the chart as.
    """

    # Filter out 'eye', 'stance', and 'unconfident'
    filtered_data = {k: v for k, v in data_dict.items() if k not in ['eye', 'stance', 'unconfident']}

    # Arabic labels
    arabic_labels = {
        "Sad": "حزين",
        "Happy": "سعيد",
        "Natural": "طبيعي",
        "Stress": "خوف"
    }
    filtered_data_arabic = {arabic_labels.get(k, k): v for k, v in filtered_data.items()}

    # Reshape and preprocess bar labels
    reshaped_labels = {arabic_reshaper.reshape(label): value for label, value in filtered_data_arabic.items()}
    bidi_labels = {get_display(label): value for label, value in reshaped_labels.items()}

    # Create the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(bidi_labels.keys(), bidi_labels.values())

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                str(f"{height:.2f}"), ha='center', va='bottom')
    colors = ['#070045', '#505ab8', '#ffaa6a', '#aaace3']

    # Set colors for each bar
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    # Set title and labels (Arabic)
    reshaped_title = arabic_reshaper.reshape('حالة المتحدث العاطفية')
    bidi_title = get_display(reshaped_title)
    ax.set_title(bidi_title, fontsize=18)

    reshaped_x = arabic_reshaper.reshape('المشاعر')
    bidi_x = get_display(reshaped_x)
    ax.set_xlabel(bidi_x, fontsize=14)

    reshaped_y = arabic_reshaper.reshape('نسبة التكرار')
    bidi_y = get_display(reshaped_y)
    ax.set_ylabel(bidi_y, fontsize=14)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'static/{filename}')

    plt.close(fig)


app = Flask(__name__,static_folder='static')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        video_file = request.files['file']
        video_file.save(f'static/uploaded_video.mp4')
        process_video(f'static/uploaded_video.mp4', f'static/annotated_video.mp4')

        # Load JSON data from a file (replace 'your_data.json' with your file path)
        with open('results.json', 'r') as f:
            data = json.load(f)

        # Data for the graphs
        data_dict = calculate_average_confidences(data)
        print(data_dict)
        create_gauge_chart(data_dict, 'unconfident', 'مستوى الثقة', '#505ab8', 'confident.png')
        create_gauge_chart(data_dict, 'stance', 'نسبة اعتدال الوقفة', '#db5959', 'stance.png')
        create_gauge_chart(data_dict, 'eye', 'نسبة توزيع النظر', '#ffaa6a', 'eye.png')
        create_bar_chart(data_dict, 'emotion.png')

        # Get user input for video path
        video_path = f'static/uploaded_video.mp4'

        # Set paths for audio and transcription output
        audio_output_path = f'static/extracted_audio.wav'
        transcription_path = f'static/transcription.txt'
        advice="nice\n"
        return render_template('result.html', advice=advice)


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/")
def index():
    return render_template("index.html")


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), 10, (frame_width, frame_height))
    
    frame_count = 0
    
    # To store results for JSON
    results_data = []  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 4 == 0:
            results = model1.predict(source=frame, save=False)
            model_name='Confidence_Score'
        elif frame_count % 4 == 1:
            results = model2.predict(source=frame, save=False)
            class_names = model2.names
            model_name='Eyes_Gaze'
        elif frame_count % 4 == 2:
            results = model3.predict(source=frame, save=False)
            class_names = model3.names
            model_name='Face_Expression'
        else:
            results = model4.predict(source=frame, save=False)
            class_names = model4.names
            model_name='Stand'
        
        
        if model_name=='Confidence_Score':
            if results[0].probs.top1:
                # Store results for JSON (customize as needed)
                results_data.append({
                    'model': model_name,
                    'classes': 'unconfident' if results[0].probs.top1 == 1 else 'confident',
                    'conf': results[0].probs.top1conf.item()
                })
        else:
            if results[0]:
                # Store results for JSON (customize as needed)
                results_data.append({
                    'model': model_name,
                    'classes': [class_names[int(cls_idx)] for cls_idx in results[0].boxes.cls],
                    'conf': results[0].boxes.conf.tolist(),
                })
        
        annotated_frame = results[0].plot()  
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    
    # Save results to JSON
    with open('results.json', 'w') as f:
        json.dump(results_data, f, indent=4)

    # Run the main function
    #main(video_path, audio_output_path, transcription_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
