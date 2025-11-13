import torch
from torch import nn
from torch.autograd import Variable


class BottleneckLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, norm='batch'):
        """
        From: https://github.com/vikrant7/mobile-vod-bottleneck-lstm/blob/master/network/mvod_bottleneck_lstm1.py
        Creates a bottleneck LSTM cell
        @param input_channels: number of input channels
        @param hidden_channels: number of hidden channels
        @param kernel_size: size of the kernel for convolutions (gates)
        @param norm: normalisation to use on output gates (default: 'batch')
        """
        super(BottleneckLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = norm

        self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=kernel_size,
                           groups=self.input_channels, stride=1, padding=1)
        self.Wy = nn.Conv2d(int(self.input_channels + self.hidden_channels), self.hidden_channels, kernel_size=1)
        self.Wi = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size, 1, 1,
                            groups=self.hidden_channels, bias=False)
        self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # In BottleneckLSTMCell.__init__
        if norm == 'batch':
            # 替换为 BatchNorm2d
            self.norm_wbi = nn.BatchNorm2d(self.hidden_channels)
            self.norm_wbf = nn.BatchNorm2d(self.hidden_channels)
            self.norm_wbc = nn.BatchNorm2d(self.hidden_channels)
            self.norm_wbo = nn.BatchNorm2d(self.hidden_channels)
        elif norm == 'layer':
            #
            # 在这里加回原始的 LayerNorm 实现
            #
            # 示例 (我不确定原始代码是怎么写的，但它可能长这样):
            # 警告: LayerNorm在RNN中通常作用于 (B, C, H, W) 的 (C, H, W)
            # 这需要知道 H 和 W，很可能是在 forward 中动态获取
            # 或者，更常见的是，它使用 GroupNorm 来模拟
            # self.norm_wbi = nn.GroupNorm(1, self.hidden_channels)
            # self.norm_wbf = nn.GroupNorm(1, self.hidden_channels)
            # self.norm_wbc = nn.GroupNorm(1, self.hidden_channels)
            # self.norm_wbo = nn.GroupNorm(1, self.hidden_channels)
            #
            # 如果你没有原始代码，并且不确定如何实现，
            # 最好是报错来提醒你自己：
            raise ValueError(f"BottleneckLSTMCell 不支持 'layer' norm，但配置要求使用它。")
        else:
            # norm is None or unrecognised
            pass

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialized bias of the cell (default to 1)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, x, h, c):
        """
        Forward pass Bottleneck LSTM cell
        @param x: input tensor with shape (B, C, H, W)
        @param h: hidden states
        @param c: cell states
        @return: new hidden states and new cell states
        """
        x = self.W(x)
        # Concat "gate": concatenate input and hidden layers
        y = torch.cat((x, h), 1)
        # Bottleneck gate: reduce to hidden layer size
        i = self.Wy(y)
        b = self.Wi(i)  # depth wise 3*3

        # Input gate
        if self.norm is not None:
            ci = self.sigmoid(self.norm_wbi(self.Wbi(b)))
        else:
            ci = self.sigmoid(self.Wbi(b))

        # Forget gate
        if self.norm is not None:
            cf = self.sigmoid(self.norm_wbf(self.Wbf(b)))
        else:
            cf = self.sigmoid(self.Wbf(b))

        # Multiply forget gate with cell state + add output of
        # input gate multiplied by output of the conv after bottleneck gate
        if self.norm is not None:
            cc = cf * c + ci * self.relu(self.norm_wbc(self.Wbc(b)))
        else:
            cc = cf * c + ci * self.relu(self.Wbc(b))

        # Output gate
        if self.norm is not None:
            co = self.sigmoid(self.norm_wbo(self.Wbo(b)))
        else:
            co = self.sigmoid(self.Wbo(b))

        ch = co * self.relu(cc)
        return ch, cc

    @staticmethod
    def init_hidden(batch_size, hidden, shape):
        # Mandatory to specify cuda here as Pytorch Lightning doesn't do it automatically for new tensors
        if torch.cuda.is_available():
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
        else:
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        return h_init, c_init


class BottleneckLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, norm='batch'):
        """
        Single layer Bottleneck LSTM cell
        @param input_channels: number of input channels of the cell
        @param hidden_channels: number of hidden channels of the cell
        @param norm: normalisation to use (default: 'batch')
        """
        super(BottleneckLSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)

        self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels, norm=norm)

    def forward(self, inputs, h, c):
        """
        Forward pass Bottleneck LSTM layer
        If stateful LSTM h and c must be None. Else they must be Tensor.
        @param inputs: input tensor
        @param h: hidden states (if None, they are automatically initialised)
        @param c: cell states (if None, they are automatically initialised)
        @return: new hidden and cell states
        """
        if h is None and c is None:
            h, c = self.cell.init_hidden(batch_size=inputs.shape[0], hidden=self.hidden_channels,
                                         shape=(inputs.shape[-2], inputs.shape[-1]))
        new_h, new_c = self.cell(inputs, h, c)
        return new_h, new_c
