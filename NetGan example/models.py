
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class Generator(nn.Module):
    """
    Класс генератора, использующего последовательность LSTM-клеток для преобразования латентного пространства в выборки данных.
    """

    def __init__(self, H_inputs, H, z_dim, N, rw_len, temp):
        """
        Конструктор класса генератора.

        Параметры:
        ----------
        H_inputs : int
            Размерность входных данных.
        H : int
            Размерность скрытого слоя.
        z_dim : int
            Размерность латентного пространства.
        N : int
            Количество узлов (необходимо для проекции вверх-вниз).
        rw_len : int
            Длина последовательности (количество клеток LSTM).
        temp : float
            Температура для мягкого максимума Гумбеля.
        """
        super(Generator, self).__init__()
        # Линейный слой для промежуточного представления латентного пространства
        self.intermediate = nn.Linear(z_dim, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate.weight)
        torch.nn.init.zeros_(self.intermediate.bias)
        
        # Линейные слои для инициализации скрытых состояний
        self.c_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up.weight)
        torch.nn.init.zeros_(self.c_up.bias)
        
        self.h_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up.weight)
        torch.nn.init.zeros_(self.h_up.bias)
        
        # Клетка LSTM
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)
        
        # Проекция вверх и вниз
        self.W_up = nn.Linear(H, N).type(torch.float64)
        self.W_down = nn.Linear(N, H_inputs).type(torch.float64)
        
        # Остальные параметры модели
        self.rw_len = rw_len
        self.temp = temp
        self.H = H
        self.latent_dim = z_dim
        self.N = N
        self.H_inputs = H_inputs

    def forward(self, latent, inputs, device='cuda'):
        """
        Прямой проход через генератор.

        Параметры:
        ----------
        latent : torch.Tensor
            Латентный вектор.
        inputs : torch.Tensor
            Входные данные.
        device : str, optional
            Устройство вычислений ('cpu' или 'cuda'). По умолчанию 'cuda'.

        Возвращает:
        -----------
        torch.Tensor
            Генерируемые данные.
        """
        # Преобразование латентного вектора в промежуточное представление
        intermediate = torch.tanh(self.intermediate(latent))
        # Инициализация скрытых состояний
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        out = []
        for _ in range(self.rw_len):
            # Пропускаем через клетку LSTM
            hh, cc = self.lstmcell(inputs, hc)
            hc = (hh, cc)
            # Поднимаемся вверх и получаем выборку
            h_up = self.W_up(hh)
            h_sample = self.gumbel_softmax_sample(h_up, self.temp, device)
            # Опускаемся вниз и обновляем вход
            inputs = self.W_down(h_sample)
            out.append(h_sample)
        return torch.stack(out, dim=1)

    def sample_latent(self, num_samples, device):
        """
        Генерация случайных образцов из нормального распределения.

        Параметры:
        ----------
        num_samples : int
            Количество образцов.
        device : str
            Устройство вычислений.

        Возвращает:
        -----------
        torch.Tensor
            Случайные образцы.
        """
        return torch.randn(num_samples, self.latent_dim).type(torch.float64).to(device)

    def sample(self, num_samples, device):
        """
        Генерация данных путем прямого прохода через модель.

        Параметры:
        ----------
        num_samples : int
            Количество генерируемых образцов.
        device : str
            Устройство вычислений.

        Возвращает:
        -----------
        torch.Tensor
            Генерируемые данные.
        """
        noise = self.sample_latent(num_samples, device)
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(device)
        generated_data = self(noise, input_zeros, device)
        return generated_data

    def sample_discrete(self, num_samples, device):
        """
        Генерация дискретных данных.

        Параметры:
        ----------
        num_samples : int
            Количество образцов.
        device : str
            Устройство вычислений.

        Возвращает:
        -----------
        np.array
            Дискретизированные данные.
        """
        with torch.no_grad():
            proba = self.sample(num_samples, device)
        return np.argmax(proba.cpu().numpy(), axis=2)

    def sample_gumbel(self, logits, eps=1e-20):
        """
        Генерация шума Гумбеля.

        Параметры:
        ----------
        logits : torch.Tensor
            Логиты.
        eps : float, optional
            Малое значение для численной стабильности. По умолчанию 1e-20.

        Возвращает:
        -----------
        torch.Tensor
            Шум Гумбеля.
        """
        U = torch.rand(logits.shape, dtype=torch.float64)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device, hard=True):
        """
        Выборка с использованием мягкого максимума Гумбеля.

        Параметры:
        ----------
        logits : torch.Tensor
            Логиты.
        temperature : float
            Температура.
        device : str
            Устройство вычислений.
        hard : bool, optional
            Если True, возвращает жесткое распределение. По умолчанию True.

        Возвращает:
        -----------
        torch.Tensor
            Образец.
        """
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(device)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=1)
        if hard:
            y_hard = torch.max(y, 1, keepdim=True)[0].eq(y).type(torch.float64).to(device)
            y = (y_hard - y).detach() + y
        return y

    def init_hidden(self, batch_size):
        """
        Инициализация нулевых скрытых состояний.

        Параметры:
        ----------
        batch_size : int
            Размер батча.

        Возвращает:
        -----------
        torch.Tensor
            Нулевые скрытые состояния.
        """
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)


class LSTMCell(nn.Module):
    """
    Клетка LSTM.
    """

    def __init__(self, input_size, hidden_size):
        """
        Конструктор клетки LSTM.

        Параметры:
        ----------
        input_size : int
            Размерность входных данных.
        hidden_size : int
            Размерность скрытого слоя.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Объединенный линейный слой для всех ворот
        self.cell = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)

    def forward(self, x, hidden):
        """
        Прямой проход через клетку LSTM.

        Параметры:
        ----------
        x : torch.Tensor
            Входные данные.
        hidden : tuple
            Скрытое состояние и память.

        Возвращает:
        -----------
        tuple
            Новое скрытое состояние и новая память.
        """
        hx, cx = hidden
        gates = torch.cat((x, hx), dim=1)
        gates = self.cell(gates)
        # Разделение на отдельные ворота
        ingate, cellgate, forgetgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(torch.add(forgetgate, 1.0))
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        # Обновление памяти
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


class Discriminator(nn.Module):
    """
    Класс дискриминатора, используемый для различения реальных и синтетических данных.
    """

    def __init__(self, H_inputs, H, N, rw_len):
        """
        Конструктор класса дискриминатора.

        Параметры:
        ----------
        H_inputs : int
            Размерность входных данных.
        H : int
            Размерность скрытого слоя.
        N : int
            Количество узлов (необходимо для проекции вверх-вниз).
        rw_len : int
            Длина последовательности (количество клеток LSTM).
        """
        super(Discriminator, self).__init__()
        # Линейный слой для проекции вниз
        self.W_down = nn.Linear(N, H_inputs, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        # Клетка LSTM
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)
        # Выходной линейный слой
        self.lin_out = nn.Linear(H, 1, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        torch.nn.init.zeros_(self.lin_out.bias)
        # Остальные параметры модели
        self.H = H
        self.N = N
        self.rw_len = rw_len
        self.H_inputs = H_inputs

    def forward(self, x):
        """
        Прямой проход через дискриминатор.

        Параметры:
        ----------
        x : torch.Tensor
            Входные данные.

        Возвращает:
        -----------
        torch.Tensor
            Предсказанные вероятности подлинности данных.
        """
        x = x.view(-1, self.N)
        xa = self.W_down(x)
        xa = xa.view(-1, self.rw_len, self.H_inputs)
        hc = self.init_hidden(xa.size(0))
        for i in range(self.rw_len):
            hc = self.lstmcell(xa[:, i, :], hc)
        out = hc[0]
        pred = self.lin_out(out)
        return pred

    def init_inputs(self, num_samples):
        """
        Инициализация входных данных.

        Параметры:
        ----------
        num_samples : int
            Количество образцов.

        Возвращает:
        -----------
        torch.Tensor
            Инициализированные входные данные.
        """
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        """
        Инициализация скрытых состояний.

        Параметры:
        ----------
        num_samples : int
            Количество образцов.

        Возвращает:
        -----------
        tuple
            Инициализированные скрытые состояния.
        """
        weight = next(self.parameters()).data
        return (
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64),
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64)
        )