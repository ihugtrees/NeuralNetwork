from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(nn.Module):
    def __init__(self, args, n_layers=1):
        super(Predictor, self).__init__()

        self.input_size = args.input_size
        self.seq_size = args.seq_size
        self.embedding_dim = args.pred_hidden_state_size
        self.lstm_enc = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.embedding_dim,
            num_layers=n_layers,
            batch_first=True  # (batch, seq, feature)
        )
        self.lstm_dec = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Linear(self.embedding_dim, args.input_size)

    def forward(self, x):
        x = x.unsqueeze(dim=2)
        z, (hs, cs) = self.lstm_enc(x)
        dec_hidden = hs.view(-1, 1, self.embedding_dim*self.factor).repeat(1, self.seq_size, 1)
        z, (hidden_state, cell_state) = self.lstm_dec(dec_hidden)
        return self.head(z)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_layers = 1
        self.batch_size = args.batch_size
        self.seq_size = args.seq_size
        self.input_size = args.input_size
        self.embedding_dim = args.hidden_state_size
        bidirectional = args.bidirectional if 'bidirectional' in args else False
        self.factor = 2 if bidirectional else 1
        self.lstm_enc = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.embedding_dim,
            num_layers=self.n_layers,
            bidirectional=bidirectional,
            batch_first=True  # (batch, seq, feature)
        )

    def forward(self, x):
        x = x.reshape(self.batch_size, self.seq_size, self.input_size)
        z, (hs, cs) = self.lstm_enc(x)
        dec_hidden = hs.view(-1, 1, self.embedding_dim*self.factor).repeat(1, self.seq_size, 1)
        return x, (dec_hidden, cs)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        bidirectional = args.bidirectional if 'bidirectional' in args else False
        self.n_layers = 1
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.embedding_dim = args.hidden_state_size
        self.factor = 2 if bidirectional else 1
        self.lstm_dec = nn.LSTM(
            input_size=self.embedding_dim*self.factor,
            hidden_size=self.embedding_dim,
            bidirectional=bidirectional,
            num_layers=self.n_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.embedding_dim*self.factor, 1)
        # self.output_layer2 = nn.Linear(self.embedding_dim, self.input_size)

    def forward(self, x):
        z, (hidden_state, cell_state) = self.lstm_dec(x)
        return self.output_layer(z), hidden_state


class LSTM_Autoencoder(nn.Module):
    def __init__(self, args):
        super(LSTM_Autoencoder, self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(device)
        self.decoder = Decoder(args).to(device)

    def forward(self, x):
        z, (hs, cs) = self.encoder(x)
        x_hat, hidden_state = self.decoder(hs)
        return x_hat.squeeze()


class LSTM_Autoencoder_SP500_Predict(nn.Module):
    def __init__(self, args):
        super(LSTM_Autoencoder_SP500_Predict, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.predict = Predictor(args)
        self.predication_length = args.predication_length

    def forward(self, x):
        z, (hs, cs) = self.encoder(x)
        predictions = []
        x_hat, hidden_state = self.decoder(hs)
        if self.args.part == 4:
            for i in range(self.predication_length):
                pred = self.predict(x).squeeze()
                p_0 = pred[:, 0].unsqueeze(dim=1)
                predictions.append(p_0)
                x = torch.hstack([x[:, 1:], p_0])
                if not i % 20:
                    print(f'{i}')
                pred = self.predict(input).squeeze(dim=2)
                predictions.append(pred)
                input = torch.hstack([x[:, 1:], pred])
            predictions = torch.hstack(predictions)
        else:
            predictions = torch.vstack([self.predict(i) for i in x.split(self.args.input_size)])
        return x_hat.squeeze(), predictions.squeeze()


class MNIST_LSTM_Autoencoder_Pixel(nn.Module):
    def __init__(self, args):
        super(MNIST_LSTM_Autoencoder_Pixel, self).__init__()
        self.seq_size = args.seq_size
        self.batch_size = args.batch_size
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.class_head = nn.Linear(args.hidden_state_size, len(args.classes))

    def forward(self, x):
        new_x = x.squeeze().reshape(self.batch_size, self.seq_size)
        z, (hs, cs) = self.encoder(new_x)
        x_hat, hidden_state = self.decoder(hs)
        imgs = x_hat.reshape(self.batch_size, 28, 28)
        linear = self.class_head(hidden_state.squeeze())
        return imgs, linear


class MNIST_LSTM_Autoencoder(nn.Module):
    def __init__(self, args):
        super(MNIST_LSTM_Autoencoder, self).__init__()
        self.seq_size = args.seq_size
        self.batch_size = args.batch_size
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.class_head_1 = nn.Linear(args.hidden_state_size, len(args.classes))
        self.class_head_2 = nn.Linear(len(args.classes), len(args.classes))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        z, (hs, cs) = self.encoder(x)
        x_hat, hidden_state = self.decoder(hs)
        hidden_state = self.dropout(hidden_state)
        linear = self.class_head_1(hidden_state.squeeze())
        linear = self.class_head_2(linear)
        return x_hat, linear
