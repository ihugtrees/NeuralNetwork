import argparse

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import TensorDataset
from utils import get_optimizer, plot_loss
from lstm_ae_model import LSTM_Autoencoder, LSTM_Autoencoder_SP500_Predict
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('-epochs', '--epochs', default=30)
parser.add_argument('-optimizer', '--optimizer', default='adam')
parser.add_argument('-scheduler_gamma', '--scheduler_gamma', default=0.8)
parser.add_argument('-grad_clip', '--grad_clip', default=1)
parser.add_argument('-lr', '--lr', default=1e-2)
parser.add_argument('-batch_size', '--batch_size', default=32)
parser.add_argument('-hidden_state_size', '--hidden_state_size', default=100)
parser.add_argument('-pred_hidden_state_size', '--pred_hidden_state_size', default=100)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp500_transform = transforms.Compose(
    [transforms.ToTensor()]
)


def validate_sp500(model, history, sp500_val, part=3, d=1):
    scaler = MinMaxScaler()
    val_stocks = sp500_val.index.to_list()
    sp500_val = scaler.fit_transform(sp500_val.T).T
    sp500_val = TensorDataset(torch.tensor(sp500_val))
    test_data_loader = torch.utils.data.DataLoader(sp500_val, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_iter = iter(test_data_loader)
    reconstruction_criterion = nn.MSELoss(reduction='mean')
    pred_criterion = nn.MSELoss(reduction='mean')
    val_loss = []
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for i, data in enumerate(test_iter):
            all_inputs = data[0]
            val_seq_loss = []
            x_hats = []
            if part == 3:
                for sub_seq in range(all_inputs.shape[1] // args.seq_size):
                    inputs = all_inputs[:, sub_seq * args.seq_size:(sub_seq + 1) * args.seq_size]
                    inputs = inputs.float().to(device)
                    if isinstance(model, LSTM_Autoencoder):
                        x_hat = model(inputs)
                        x_hats.append(x_hat.cpu())
                        rec_loss = reconstruction_criterion(x_hat, inputs)
                        val_seq_loss.append(rec_loss.item())
                    elif isinstance(model, LSTM_Autoencoder_SP500_Predict):
                        x_hat, preds = model(inputs)
                        pred_length = args.predication_length
                        start_pred = args.start_pred
                        vec = torch.hstack([x_hat[:, 1].unsqueeze(dim=1), preds[:, start_pred:]])
                        x_hats.append(vec.cpu())
                        rec_loss = reconstruction_criterion(x_hat, inputs)
                        pred_loss = pred_criterion(preds[:, :-start_pred], inputs[:, start_pred:])
                        loss = rec_loss + d * pred_loss
                        val_seq_loss.append(loss.item())
                val_loss.append(np.mean(val_seq_loss))
                all_preds.append(torch.hstack(x_hats))
                all_gt.append(all_inputs)
            else:
                inputs = all_inputs.float().to(device)
                if isinstance(model, LSTM_Autoencoder_SP500_Predict):
                    pred_length = args.predication_length
                    start_pred = args.start_pred
                    x_hat, preds = model(inputs[:, :start_pred])
                    vec = torch.hstack([inputs[:, :start_pred], preds])
                    x_hats.append(vec.cpu())
                    rec_loss = reconstruction_criterion(x_hat, inputs[:, :start_pred])
                    pred_loss = pred_criterion(preds, inputs[:, start_pred:])
                    loss = rec_loss + d * pred_loss
                    val_seq_loss.append(loss.item())
                val_loss.append(np.mean(val_seq_loss))
                all_preds.append(torch.hstack(x_hats))
                all_gt.append(all_inputs)
    loss = np.mean(val_loss)
    history['val'].append(loss)
    print(f"end of validation: val loss={loss}")
    return torch.vstack(all_preds), torch.vstack(all_gt), val_stocks


def train_sp500(model, datasets, part=3, d=0.5):
    sp500_train, sp500_val = datasets
    scaler = MinMaxScaler()
    train_stocks = sp500_train.index.to_list()
    val_stocks = sp500_val.index.to_list()
    sp500_train = scaler.fit_transform(sp500_train.T).T
    sp500_train = TensorDataset(torch.tensor(sp500_train))
    optimizer = get_optimizer(model, args)
    reconstruction_criterion = nn.MSELoss(reduction='mean')
    pred_criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[])
    for epoch in range(args.epochs):
        epoch_train_losses = []
        train_data_loader = torch.utils.data.DataLoader(sp500_train, batch_size=args.batch_size, shuffle=True,
                                                        drop_last=True)
        for i, data in enumerate(iter(train_data_loader)):
            if not i % 5:
                print(f'epoch {epoch}: iter {i}, train loss={np.mean(epoch_train_losses)}')
            all_inputs = data[0]
            batch_loss = []
            for sub_seq in range(all_inputs.shape[1] // args.seq_size):
                optimizer.zero_grad()
                inputs = all_inputs[:, sub_seq * args.seq_size:(sub_seq + 1) * args.seq_size]
                inputs = inputs.float().to(device)
                if isinstance(model, LSTM_Autoencoder):
                    x_hat = model(inputs)
                    loss = reconstruction_criterion(x_hat, inputs)
                    loss.backward()
                    batch_loss.append(loss.item())
                elif isinstance(model, LSTM_Autoencoder_SP500_Predict):
                    x_hat, preds = model(inputs)
                    start_pred = args.start_pred
                    rec_loss = reconstruction_criterion(x_hat, inputs)
                    pred_loss = pred_criterion(preds[:, :-start_pred], inputs[:, start_pred:])
                    loss = rec_loss + d * pred_loss
                    loss.backward()
                    batch_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            epoch_train_losses.append(np.mean(batch_loss))
        history['train'].append(np.mean(epoch_train_losses))
        print(f"end of epoch {epoch}: train loss={np.mean(epoch_train_losses)}")

        val_pred, val_truth, _ = validate_sp500(model, history, sp500_val, part=3, d=d)
        plot_signals(datasets[1], val_pred, val_truth)
        if part == 3:
            torch.save(model.state_dict(), 'model.pt')
    plot_loss(history, args)


def rescale_signal(signal, max_stock, min_stock):
    return signal * (max_stock - min_stock) + min_stock


def plot_signals(val_stocks, pred, truth):
    import matplotlib.pyplot as plt
    rand = np.random.randint(len(pred)-1)
    stock = val_stocks.iloc[rand]
    p, t = pred.detach()[rand], truth.detach()[rand]
    p = rescale_signal(p, stock.max(), stock.min())
    t = rescale_signal(t, stock.max(), stock.min())
    plt.plot(p)
    plt.plot(t)
    stock_ticker = stock.name
    plt.legend((f"prediction of stock {stock_ticker}", f"truth value of stock {stock_ticker}"))
    plt.title(f"Truth vs Pred of stock {stock_ticker}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()


def plot_googl_amzn(sp):
    googl_amzn = sp[sp.symbol.isin(['GOOGL', 'AMZN'])][['date', 'symbol', 'high']].set_index('date')
    figure = px.line(googl_amzn, color='symbol', labels={'value': 'daily maximum'},
                     title="Google vs. Amazon max value per day", )
    figure.update_layout(title={'x': 0.5})
    figure.show()


def split_stocks(price_df, per=0.2):
    temp = price_df.T.sample(frac=1)
    train, val = train_test_split(temp, test_size=per, shuffle=False)
    return train, val, temp.index.to_list()


def reconstruct_stocks(price_df):
    price_df_n = price_df
    train, val, order = split_stocks(pd.DataFrame(price_df_n))
    return order, (train, val)


def train_reconstruct(price_df, datasets):
    args.seq_size = price_df.shape[0]
    args.input_size = 1
    args.seq_size = 53  # 19 batches
    model = LSTM_Autoencoder(args).to(device)
    train_sp500(model, datasets)


def train_predict_next(price_df, datasets, d=1, part=3):
    args.input_size = 1
    if part == 4:
        args.part = 4
        seq_pred_length = datasets[1].shape[1] // 2
        args.seq_size = seq_pred_length + 1
        args.predication_length = seq_pred_length
        args.start_pred = seq_pred_length + 1
        model = LSTM_Autoencoder_SP500_Predict(args).to(device)
        weights = 'model.pt'
        model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        val_pred, val_truth, val_stocks = validate_sp500(model, dict(val=[]), datasets[1], part=4)
        plot_signals(datasets[1], val_pred, val_truth)
    elif part == 3:
        args.part = 3
        args.seq_size = 53  # 19 batches
        args.start_pred = 1
        args.predication_length = args.seq_size - args.start_pred
        model = LSTM_Autoencoder_SP500_Predict(args).to(device)
        train_sp500(model, datasets, d=d, part=3)


if __name__ == '__main__':
    # part 1:
    sp = pd.read_csv('sp_prices.csv')
    # plot_googl_amzn(sp)

    price_df = sp.pivot(index='date', columns='symbol', values='high').dropna(how='any', axis=1)
    order, datasets = reconstruct_stocks(price_df)

    # part 2:
    train_reconstruct(price_df, datasets)

    # part 3:
    train_predict_next(price_df, datasets, d=1, part=3)

    # part 4:
    train_predict_next(price_df, datasets, part=4)