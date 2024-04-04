from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": "Bearer "}

def query_huggingface_api(image_data):
    response = requests.post(API_URL, headers=headers, data=image_data)
    return response.json()

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        image_data = file.read()
        response = query_huggingface_api(image_data)
        return jsonify(response)
    
    return jsonify({"error": "Unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
