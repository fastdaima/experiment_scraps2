# Load model directly
import os 

from transformers import AutoTokenizer, AutoModelForCausalLM

token = os.environ['HF_TOKEN']

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the pre-trained language model
model_name = "gpt2"  # You can replace this with the model you want to use


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data['prompt']
        max_length = data.get('max_length', 50)

        # Generate text using the model
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        response = {'generated_text': generated_text}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
