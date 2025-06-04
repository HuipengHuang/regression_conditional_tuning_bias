import torch
import torch.nn as nn

# Define the network
class mse_model(nn.Module):
    """ Conditional mean estimator, formulated as neural net
    """

    def __init__(self,
                 in_shape,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------

        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_shape = in_shape
        self.out_shape = 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device)

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return torch.squeeze(self.base_model(x))



class all_q_model(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 quantiles,
                 in_shape,
                 hidden_size=64,
                 dropout=0.5):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        in_shape : integer, input signal dimension (p)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate

        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_quantiles),
        ).to(self.device)

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return self.base_model(x)




