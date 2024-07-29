from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

conversation_history = []

from flask import request, render_template		# newly added
import json

from flask import Flask
from flask_cors import CORS		# newly added

app = Flask(__name__)
CORS(app)	

@app.route('/chatbot', methods=['POST'])

def handle_prompt():

    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    history_string = "\n".join(conversation_history)

    # tokenizer.pretrained_vocab_files_map    ##for vocabulary

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    print(inputs)

    outputs = model.generate(**inputs)
    print(outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)
    
    return response
			# newly added

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)