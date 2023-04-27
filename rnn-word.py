import numpy as np
import torch
from torch import nn

def main():

    sentances = []
    sentance_length = 100

    # Read the data file into a string
    with open("tiny-shakespeare.txt") as f:
        input = f.read()

    # Splits the input up by spaces
    inputList = input.split() 

    # Create a dictionary of words
    words = set(inputList)

    # Create sentances which holds strings of 100 words each
    line = []
    for i in range(len(inputList) - 1):
        line.append(inputList[i])
        if (i + 1) % 100 == 0:
            sentances.append(" ".join(line))
            line = []

    
    # Set up the vocabulary. It's convenient to have dictionaries that can go both ways.
    intWord = dict(enumerate(words))
    wordInt = {word: index for index, word in intWord.items()}

    # Offset input and output sentances
    input_sequence = []
    target_sequence = []
    for i in range(len(sentances)):
        # Remove the last character for the input sequence
        line_without_first = sentances[i].split()
        del line_without_first[0]
        input_sequence.append(" ".join(line_without_first))
        # Remove the first characters for the target sequences
        line_without_last = sentances[i].split()
        del line_without_last[len(line_without_last) - 1]
        target_sequence.append(" ".join(line_without_last))
    
    # Construct the one hots. 
    # Replace all words with integers
    for i in range(len(sentances)):
        input_sequence[i] = [wordInt[word] for word in input_sequence[i].split()]
        target_sequence[i] = [wordInt[word] for word in target_sequence[i].split()]
    
    # Need vocab size to make the one-hots
    vocab_size = len(wordInt)
    batch_size = 1
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

    epochs = 10
    finalLoss = 0
    firstLoss = 0

    # TRAIN
    for epoch in range(1, epochs+1):
        for i in range(len(input_sequence)):
            optimizer.zero_grad()
            # Create the input and target tensors
            x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size, sequence_length, batch_size)).to(device)
            y = torch.Tensor(target_sequence[i]).to(device)
            output, hidden = model(x)

            lossValue = loss(output, y.view(-1).long())
            # Calculates gradient
            lossValue.backward()
            # Updates weights
            optimizer.step()
        
            if epoch == 1:
                firstLoss += lossValue.item()
            if epoch == epochs:
                finalLoss += lossValue.item()
        if epoch % 2 == 0:
            print("Epoch: {}/{}........".format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(lossValue.item()))
        if epoch == 1:
            print("First Loss: {:.4f}".format(firstLoss / len(input_sequence)))
        if epoch == epochs:
            print("\nFinal Loss: {:.4f}".format(finalLoss / len(input_sequence)))

    # Begin to output
    print(sample(model, 25, wordInt, intWord, vocab_size, device, "Queen of how you come to us"))


def create_one_hot(sequence, vocab_size, sequence_length, batch_size):
    #Tensor is of the form (batch size, sequence length, one-hot length)
    encoding = np.zeros((batch_size, sequence_length, vocab_size), dtype=np.float32)
    for i in range(batch_size):
        encoding[0, i, sequence[i]] = 1

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

# Given a sequence of words uses the model to predict the next word    
def predict(model, words, wordInt, vocab_size, intWord, device):

    wordInput = np.array([wordInt[w] for w in words])
    wordInput = create_one_hot(wordInput, vocab_size, len(wordInput), 1)
    wordInput = torch.from_numpy(wordInput).to(device)

    out, hidden = model(wordInput)

    #Get output probabilities
    prob = nn.functional.softmax(out[-1], dim=0).data

    word_index = torch.max(prob, dim=0)[1].item()

    return intWord[word_index], hidden

# Given a starting sentance uses the model to predict the words that follow up to word length out_len
def sample(model, out_len, charInt, intChar, vocab_size, device, start):
    model.eval()
    words = start.split()
    currentSize = out_len - len(start.split())
    for i in range(currentSize):
        word, hidden_state = predict(model, words, charInt, vocab_size, intChar, device)
        words.append(word)

    return ' '.join(words)


main()