"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import google.generativeai as genai

genai.configure(api_key="AIzaSyAvlxrF4P04K93bjj7LtSCJh81RzX69-xs")

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["what is machine learning"]
  },
  {
    "role": "model",
    "parts": ["## Machine Learning Explained\n\nMachine learning, often abbreviated as ML, is a subfield of artificial intelligence (AI) that focuses on enabling machines to learn from data and improve their performance on a specific task without explicit programming. In simpler terms, it's like teaching a computer to learn and adapt like humans do, but with massive amounts of data and complex algorithms.\n\nHere's a breakdown of the key aspects:\n\n**1. Learning from Data:**\n\n*   Instead of being explicitly programmed with rules, ML models are trained on large datasets. \n*   These datasets contain examples (like images, text, or numbers) and their corresponding desired outputs.\n*   The model learns patterns and relationships within the data, allowing it to make predictions or decisions on new, unseen data.\n\n**2. Types of Machine Learning:**\n\n*   **Supervised Learning:** The model learns from labeled data, where each example has a known output (e.g., classifying emails as spam or not spam).\n*   **Unsupervised Learning:** The model finds hidden patterns or structures in unlabeled data (e.g., grouping customers based on their purchase history).\n*   **Reinforcement Learning:** The model learns through trial and error by interacting with an environment and receiving rewards or penalties (e.g., training a robot to navigate a maze).\n\n**3. Applications of Machine Learning:**\n\nMachine learning is used in a vast array of applications, including:\n\n*   **Image and speech recognition:** Facial recognition, voice assistants like Siri or Alexa.\n*   **Natural language processing:** Machine translation, sentiment analysis, chatbots.\n*   **Recommendation systems:** Personalized recommendations on platforms like Netflix or Amazon.\n*   **Medical diagnosis:** Analyzing medical images, predicting patient outcomes.\n*   **Financial fraud detection:** Identifying suspicious transactions, preventing fraud.\n*   **Self-driving cars:** Interpreting sensor data, making driving decisions. \n \n**4. Benefits of Machine Learning:**\n\n*   **Automation:** Automates repetitive tasks, freeing up human time and resources.\n*   **Improved accuracy:**  Makes more accurate predictions and decisions compared to traditional methods.\n*   **Personalization:** Provides customized experiences based on individual preferences.\n*   **Discovery:** Uncovers hidden patterns and insights from complex data.\n\n**5. Challenges of Machine Learning:**\n\n*   **Data dependency:** Requires large amounts of high-quality data for training.\n*   **Bias:**  Models can perpetuate or amplify existing biases in data. \n*   **Explainability:** Understanding how models make decisions can be difficult.\n*   **Ethical considerations:** Concerns about privacy, fairness, and the impact of automation on jobs."]
  },
])

convo.send_message("what is machine learning")
print(convo.last.text)
