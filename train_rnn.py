import neuralnets as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
HIDDEN = 32
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.rnn.RNN(
            n_in=INPUT_SIZE,
            n_hidden=HIDDEN,     # rnn hidden unit
            n_out=1,
            batch_first=True,
        )
        self.out = nn.layers.Dense(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        outs, latest_state = self.rnn(x, h_state)
        outs
        return outs, h_state


rnn = RNN()
opt = nn.optim.Adam(rnn.params, lr=LR)   # optimize all cnn parameters
loss_fn = nn.losses.MSE()

h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x = np.sin(steps)[None, :, None]    # (batch, time_step, input_size)
    y = np.cos(steps)[None, :, None]    # (batch, time_step, input_size)

    predictions, h_state = rnn(x, h_state)   # rnn output
    loss = loss_fn(predictions, y)
    rnn.backward(loss)
    opt.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, predictions.data.flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()