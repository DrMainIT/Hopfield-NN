from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

len_of_input = 4

# Add a global variable to store the weight matrix
current_weights = [[0] * len_of_input for _ in range(len_of_input)]

class Network:
    def __init__(self, weights):
        self.neurons = [Neuron(weights[i]) for i in range(len(weights))]
        self.fields = len_of_input
        self.output = [False] * self.fields

    def threshold(self, k):
        return k >= 0

    def activation(self, pattern):
        for i in range(self.fields):
            if i < len(self.neurons):  # Ensure index is valid
                self.neurons[i].activation = self.neurons[i].act(pattern)
                self.output[i] = self.threshold(self.neurons[i].activation)

class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.activation = 0

    def act(self, pattern):
        activation_value = sum(self.weights[i] if pattern[i] else 0 for i in range(len_of_input))
        return activation_value

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_network():
    data = request.json
    pattern = data['input_pattern']

    # Debugging: Print current weights and input pattern
    print("Current Weights:", current_weights)
    print("Input Pattern:", pattern)

    net = Network(current_weights)  # Use the updated weight matrix
    net.activation(pattern)

    return jsonify(output=net.output)

@app.route('/train', methods=['POST'])
def train():
    global current_weights  # Use the global variable to store weights
    data = request.json
    input_pattern = data['input_pattern']

    # Compute the Hebbian update based on the input pattern
    bi = [(1 if x == "1" else -1) for x in input_pattern]
    weight_update = [[bi[row] * bi[col] for col in range(len_of_input)] for row in range(len_of_input)]

    # Ensure no self-activation (subtract 1 from the diagonal)
    for x in range(len_of_input):
        weight_update[x][x] -= 1

    # Update the weight matrix
    for row in range(len_of_input):
        for col in range(len_of_input):
            current_weights[row][col] += weight_update[row][col]

    # Return the updated weight matrix
    return jsonify(weight_matrix=current_weights)

# Add a route to clear the weight matrix
@app.route('/clear', methods=['POST'])
def clear():
    global current_weights
    current_weights = [[0] * len_of_input for _ in range(len_of_input)]
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
