from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        if torch.isnan(scale).any():
            # Replace NaN values with a default value
            min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
            scale = torch.where(torch.isnan(scale), min_scale, scale)

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
        scale = torch.max(scale, min_scale)
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()
            
            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None

def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False):
    if qparams:
        if qparams.num_bits:
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace)

    return x


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            return q_input

class QLinear(nn.Linear):
    """docstring for QLinear."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quantize_input = QuantMeasure(shape_measure=(1,1,1), flatten_dims=(1,-1))
        self.num_bits = num_bits
        
    def forward(self, input):
        if self.num_bits == 0:
            output = F.linear(input, self.weight, self.bias)
            return output
        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits,
                flatten_dims=(-1,))
        else:
            qbias = None
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits, flatten_dims=(1,-1),
                                           reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)
        qinput = self.quantize_input(input, self.num_bits)
        output = F.linear(qinput, qweight, qbias)
        return output

def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    m.total_ops += calculate_linear(total_mul, num_elements)


if __name__ == '__main__':
    # Define input parameters
    in_features = 5
    out_features = 3
    batch_size = 2

    # Create an instance of QLinear
    qlinear = QLinear(in_features, out_features).cuda()

    # Generate some random input
    input_data = torch.rand(batch_size, in_features).cuda()

    # Define the number of bits for quantization
    num_bits = 8

    # Forward pass through the QLinear layer
    output = qlinear(input_data, num_bits=num_bits)

    # Print the output
    print("Output after quantization:")
    print(output)