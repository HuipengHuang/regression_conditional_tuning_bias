import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, n_binary=10, n_continuous=90, embedding_dim=4, dropout_rate=0.1):
        super(SimpleNN, self).__init__()
        self.n_binary = n_binary
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(2, embedding_dim)  # Embedding for binary variables

        # Calculate the total input dimension after embedding binary variables
        total_input_dim = n_binary * embedding_dim + n_continuous

        self.layer1 = nn.Linear(total_input_dim, 20)
        self.norm1 = nn.LayerNorm(20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first ReLU

        self.layer2 = nn.Linear(20, 10)
        self.norm2 = nn.LayerNorm(10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second ReLU

        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        # Split the input into binary and continuous parts
        x_binary = x[:, :self.n_binary].long()  # Assuming the first n_binary are binary
        x_continuous = x[:, self.n_binary:]  # The rest are continuous

        # Embedding the binary variables
        x_binary = self.embedding(x_binary)  # Shape [batch_size, n_binary, embedding_dim]
        x_binary = x_binary.view(x_binary.shape[0], -1)  # Flatten the embeddings

        # Concatenate the embedded binary variables with the continuous variables
        x = torch.cat([x_binary, x_continuous], dim=1)

        # Pass through the network
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)  # Apply dropout after the first ReLU

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)  # Apply dropout after the second ReLU

        x = self.layer3(x)
        return x.squeeze()