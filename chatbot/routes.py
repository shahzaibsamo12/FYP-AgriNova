from flask import Blueprint, jsonify, request
from core.utils import get_llm_response

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/api/chat/ask', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query')
    language = data.get('language', 'en')

    if not user_query:
        return jsonify({"response": "Please provide a query."}), 400

    # Define a concise system role for structured Markdown responses
    system_role = (
        "You are AgriNova Assistant, an agricultural expert. "
        "Use Markdown for formatting. Use BOLD HEADERS (e.g., **Heading**) and BULLET POINTS. "
        "Keep responses very SHORT, structured, and easy to read. "
        "Focus only on direct answers about farming, crops, and soil. "
    )
    
    if language == 'ur':
        system_role += (
            "Provide the response in Urdu script. "
            "Use very SIMPLE, EASY-TO-UNDERSTAND Urdu words that farmers can read without difficulty. "
            "Avoid difficult or literary words. Use common everyday Urdu. "
            "Use Markdown for structure with BOLD HEADERS and BULLET POINTS."
        )
    else:
        system_role += "Provide the response in English. Use Markdown for structure."



    response = get_llm_response(user_query, system_role=system_role)

    return jsonify({
        "response": response
    })

