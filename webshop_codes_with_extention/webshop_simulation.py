from flask import Flask, request, jsonify, render_template
import json
import random
import os

app = Flask(__name__)

# Path to the data folder
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

# Load data from JSON files
with open(os.path.join(DATA_FOLDER, "items_shuffle.json")) as f:
    PRODUCTS = json.load(f)

with open(os.path.join(DATA_FOLDER, "items_ins_v2.json")) as f:
    ATTRIBUTES = json.load(f)

with open(os.path.join(DATA_FOLDER, "items_human_ins.json")) as f:
    HUMAN_INSTRUCTIONS = json.load(f)

# Simulated sessions
SESSIONS = {}

@app.route('/')
def home():
    """Home page with a search bar."""
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def start_session():
    """Start a new session."""
    session_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz1234567890', k=10))
    SESSIONS[session_id] = {'score': 0}
    return jsonify({'session_id': session_id})

@app.route('/search', methods=['GET', 'POST'])
def search_products():
    """Enhanced product search with weighted filtering."""
    if request.method == 'POST':
        data = request.json
        query = data.get('query', '').lower()
        session_id = data.get('session_id')
    else:  # Handle GET
        query = request.args.get('query', '').lower()
        session_id = request.args.get('session_id')
    if session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session'}), 400

    # Ranking logic: Name > Full Description > Small Description
    def calculate_relevance(product, query):
        score = 0
        if query in product.get('name', '').lower():
            score += 5  # High weight for name match
        if query in product.get('full_description', '').lower():
            score += 3  # Medium weight for description match
        if any(query in desc.lower() for desc in product.get('small_description', [])):
            score += 1  # Low weight for small description match
        return score

    # Filter and rank products
    results = []
    for product in PRODUCTS:
        relevance_score = calculate_relevance(product, query)
        if relevance_score > 0:
            results.append({'product': product, 'score': relevance_score})

    # Shuffle results to simulate randomness
    random.shuffle(results)

    # Sort products by relevance score (descending order)
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Include scores in the response
    return jsonify({
        'products': [{'product': res['product'], 'score': res['score']} for res in results],
        'total_results': len(results)
    })

@app.route('/buy', methods=['POST'])
def buy_product():
    """Simulate buying a product."""
    data = request.json
    session_id = data.get('session_id')
    product_name = data.get('product_name')
    product_score = data.get('product_score')  # Expect score from the client

    if session_id not in SESSIONS:
        return jsonify({'error': 'Invalid session'}), 400

    # Validate product existence
    for product in PRODUCTS:
        if product['name'] == product_name:
            # Update the session's total score using the provided score
            SESSIONS[session_id]['score'] += product_score
            return jsonify({
                'score': product_score,
                'total_score': SESSIONS[session_id]['score']
            })
    
    return jsonify({'error': 'Product not found'}), 404



if __name__ == '__main__':
    app.run(port=3000)
