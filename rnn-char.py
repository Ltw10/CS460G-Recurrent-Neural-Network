import numpy as np
import torch
from torch import nn

def main():

    sentances = []
    sentance_length = 100

    # Tokenize data file by every 100 characters
    with open("tiny-shakespeare.txt") as f:
        while True:
            line = f.read(sentance_length)
            if not line:
                break
            else:
                sentances.append(line)

    del sentances[len(sentances) - 1]
    

    # Extract all characters from sentances
    characters = set(''.join(sentances))
    
    # Set up the vocabulary. It's convenient to have dictionaries that can go both ways.
    intChar = dict(enumerate(characters))
    charInt = {character: index for index, character in intChar.items()}
    
    # No padding needed here becasue because it was already done in preprocessing

    # Offset input and output sentances
    input_sequence = []
    target_sequence = []
    for i in range(len(sentances)):
        # Remove the last character for the input sequence
        input_sequence.append(sentances[i][:-1])
        # Remove the first characters for the target sequences
        target_sequence.append(sentances[i][1:])
    
    # Construct the one hots. 
    # Replace all characters with integers
    for i in range(len(sentances)):
        input_sequence[i] = [charInt[character] for character in input_sequence[i]]
        target_sequence[i] = [charInt[character] for character in target_sequence[i]]
    
    # Need vocab size to make the one-hots
    vocab_size = len(charInt)
    batch_size = len(sentances)
    sequence_length = len(input_sequence[0])

    # Check if cuda GPU can be used to run the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Create the neural network model!
    model = RNNModel(input_size=vocab_size, output_size=vocab_size, hidden_size=100, num_layers=1, device=device)
    model.to(device)

    # Define Loss
    loss = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 1000

    # Create the input and target tensors
    x = torch.from_numpy(create_one_hot(input_sequence, vocab_size, sequence_length, batch_size)).to(device)
    y = torch.Tensor(target_sequence).to(device)

    # TRAIN
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        output, hidden = model(x)

        lossValue = loss(output, y.view(-1).long())
        # Calculates gradient
        lossValue.backward()
        # Updates weights
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch: {}/{}........".format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(lossValue.item()))
        if epoch == epochs:
            print("\nFinal Loss: {:.4f}".format(lossValue.item()))

    # Begin to output
    print(sample(model, 50, charInt, intChar, vocab_size, device, "Queen"))


def create_one_hot(sequence, vocab_size, sequence_length, batch_size):
    #Tensor is of the form (batch size, sequence length, one-hot length)
    encoding = np.zeros((batch_size, sequence_length, vocab_size), dtype=np.float32)
    for i in range(batch_size):
        for j in range(sequence_length):
            encoding[i, j, sequence[i][j]] = 1

    return encoding

# PyTorch class to contain the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        #Define the network!
        #Batch first defines where the batch parameter is in the tensor
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize the hidden state for first input
        hidden_state = self.init_hidden(batch_size).to(self.device)

        # Passing in the input and hidden state into the model and getting outputs
        output, hidden_state = self.rnn(x, hidden_state)

        # Reshaping outputs so that it can pass into fully connected layer
        output = output.contiguous().view(-1, self.hidden_size)

        output = self.fc(output)

        return output, hidden_state
    
    def init_hidden(self, batch_size):
        #Row, Batch, Column
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
    
# Given a sequence of characters uses the model to predict the next character
def predict(model, character, charInt, vocab_size, intChar, device):

    characterInput = np.array([[charInt[c] for c in character]])
    characterInput = create_one_hot(characterInput, vocab_size, characterInput.shape[1], 1)
    characterInput = torch.from_numpy(characterInput).to(device)

    out, hidden = model(characterInput)

    #Get output probabilities
    prob = nn.functional.softmax(out[-1], dim=0).data

    character_index = torch.max(prob, dim=0)[1].item()

    return intChar[character_index], hidden

# Given a starting word uses the model to predict the characters that follow up to length out_len
def sample(model, out_len, charInt, intChar, vocab_size, device, start):
    model.eval()
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters, charInt, vocab_size, intChar, device)
        characters.append(character)

    return ''.join(characters)


main()