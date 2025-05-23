from flask import Flask, request, jsonify
import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
import os
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
load_dotenv()
# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-2.0-flash")
model_vector = SentenceTransformer('all-MiniLM-L6-v2')

# Prompt for image text extraction
IMAGE_EXTRACT_PROMPT = '''Extract and transcribe all visible text and mathematical formulas from the provided image with 100% accuracy.
Important Requirements:
- Maintain the original order and grouping as shown in the image
- Preserve diacritical marks for Vietnamese or other accented text
- If any part of the text or formula is unclear or unreadable, indicate with [unreadable]
- Output in JSON format as a list of objects, each object has:
  { "type": "text", "value": "The extracted natural text" }
  or
  { "type": "formula", "value": "The formula strictly in LaTeX syntax" }
'''

# Default AI Solution Prompt
DEFAULT_AI_SOLUTION_PROMPT = '''You are an educational assistant tasked with providing detailed solutions to mathematical and scientific problems. The user will provide a question or problem, and you need to:

1. Analyze the question to identify the topic and key concepts involved
2. Provide a clear, step-by-step solution using proper mathematical notation
3. Explain the reasoning behind each step in simple terms
4. Include relevant formulas and their applications
5. Summarize the answer and key takeaways at the end

Format your response using HTML and LaTeX (inside \\( \\) for inline and \\[ \\] for display equations) to ensure proper formatting of mathematical expressions.

Make your explanation educational and helpful for students trying to understand the concept, not just getting the answer.'''

def extract_text_from_image(image_data):
    """Extract text and formulas from an image using Gemini API"""
    try:
        image_parts = [
            {'mime_type': 'image/jpeg', 'data': image_data},
            {'text': IMAGE_EXTRACT_PROMPT}
        ]

        response = model.generate_content(
            image_parts,
            generation_config={
                'top_k': 32,
                'top_p': 0.95,
                'temperature': 0.2,
                'max_output_tokens': 2048
            }
        )
        return response.text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def get_ai_solution(query, custom_prompt=None):
    """Get AI solution for a query using Gemini API"""
    try:
        # Use custom prompt if provided, otherwise use default
        prompt = custom_prompt if custom_prompt and custom_prompt.strip() else DEFAULT_AI_SOLUTION_PROMPT
        
        # Combine prompt with query
        full_prompt = f"{prompt}\n\nProblem: {query}\n\nSolution:"
        
        response = model.generate_content(
            full_prompt,
            generation_config={
                'top_k': 32,
                'top_p': 0.95,
                'temperature': 0.3,
                'max_output_tokens': 4096
            }
        )
        return response.text
    except Exception as e:
        print(f"Error getting AI solution: {e}")
        return None

def get_top_5_similar_questions(question):
    """Lấy 5 câu hỏi gần nhất trong database theo cosine similarity"""
    # Encode câu hỏi nhận được
    vector_question = model_vector.encode([question])[0]

    # Kết nối vào database MySQL
    connection = mysql.connector.connect(
        host='localhost',
        port=3307,
        user='root',
        password='1111',
        database='snap_solve'
    )

    cursor = connection.cursor()
    # Truy vấn toàn bộ vector và id
    cursor.execute("SELECT id, vector FROM assignment")
    data = cursor.fetchall()

    # Mảng chứa các vector từ DB
    vectors = []
    ids = []

    # Giải mã JSON và phục hồi vector
    for id, vector_data in data:
        vector_db = json.loads(vector_data)
        quantized_vec = np.array(vector_db['quantized'])
        max_val = vector_db['max_val']

        # Phục hồi vector ban đầu
        vector_restored = quantized_vec * max_val / 255.0
        vectors.append(vector_restored)
        ids.append(id)

    # Tính cosine similarity
    similarities = cosine_similarity([vector_question], vectors)[0]

    # Ghép với id và sắp xếp theo mức độ tương đồng
    results = sorted(zip(ids, similarities), key=lambda x: x[1], reverse=True)[:5]

    # Đóng kết nối
    cursor.close()
    connection.close()

    return results

@app.route('/extract-text', methods=['POST'])
def extract_text():
    """Endpoint to extract text from uploaded image"""
    if 'image' not in request.files:
        print("hiep")
        return jsonify({
            'success': False,
            'message': 'No image provided'
        }), 400
    
    file = request.files['image']
    try:
        # Convert image to base64
        img = Image.open(file.stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format='JPEG')
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Extract text using Gemini
        extracted_text = extract_text_from_image(image_data)
        
        if extracted_text:
            # Process the extracted text
            if "```json" in extracted_text:
                extracted_text = extracted_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extracted_text:
                extracted_text = extracted_text.split("```")[1].split("```")[0].strip()
            
            try:
                parsed_data = json.loads(extracted_text)
                
                # Format the output
                question = ""
                for item in parsed_data:
                    if item["type"] == "text":
                        question += item["value"] + " "
                    elif item["type"] == "formula":
                        question += f"\\({item['value']}\\) "
                
                question = question.strip()

                # Lấy ra 5 câu hỏi gần nhất
                top_5_similar = get_top_5_similar_questions(question)

                return jsonify({
                    'success': True,
                    'question': question,
                    'similar_questions': [
                        {'id': id, 'similarity': round(score, 4)} for id, score in top_5_similar
                    ]
                })
            except json.JSONDecodeError as e:
                return jsonify({
                    'success': False,
                    'message': f'Failed to parse JSON response: {str(e)}',
                    'raw_text': extracted_text
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to extract text from image'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500


@app.route('/search-by-text', methods=['POST'])
def search_by_text():
    """Endpoint to search similar questions by text query"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'message': 'No query provided'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'message': 'Query cannot be empty'
            }), 400
        
        # Get top 5 similar questions
        top_5_similar = get_top_5_similar_questions(query)
        
        return jsonify({
            'success': True,
            'query': query,
            'similar_questions': [
                {'id': id, 'similarity': round(score, 4)} for id, score in top_5_similar
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing text search: {str(e)}'
        }), 500

@app.route('/ai-solution', methods=['POST'])
def ai_solution():
    """Endpoint to get AI solution for a question"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'message': 'No query provided'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'message': 'Query cannot be empty'
            }), 400
        
        # Get custom prompt if provided
        custom_prompt = data.get('prompt')
        
        # Get AI solution
        solution = get_ai_solution(query, custom_prompt)
        
        if solution:
            return jsonify({
                'success': True,
                'query': query,
                'solution': solution
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to get AI solution'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing AI solution request: {str(e)}'
        }), 500
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)