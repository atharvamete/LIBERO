import torch
import torch.nn as nn
import numpy as np
from vector_quantize_pytorch import FSQ, ResidualFSQ, VectorQuantize, ResidualVQ

class Quantizer(nn.Module):
    def __init__(self, quantizer_type='vq', **kwargs):
        super(Quantizer, self).__init__()
        if quantizer_type == 'fsq':
            self.quantizer = FSQ(dim=kwargs['dim'],levels=kwargs['levels'])
            self.total_codes = np.prod(kwargs['levels'])
            self.indices_to_codes = self.quantizer.indices_to_codes
        elif quantizer_type == 'residual_fsq':
            self.quantizer = ResidualFSQ(dim=kwargs['dim'], levels=kwargs['levels'], num_quantizers=kwargs['num_quantizers'])
            self.total_codes = np.prod(kwargs['levels'])
            self.indices_to_codes = self.quantizer.get_output_from_indices
        elif quantizer_type == 'vq':
            self.quantizer = VectorQuantize(dim=kwargs['dim'], codebook_size=kwargs['codebook_size'])
            self.total_codes = kwargs['codebook_size']
            self.indices_to_codes = self.quantizer.get_output_from_indices
        elif quantizer_type == 'residual_vq':
            self.quantizer = ResidualVQ(dim=kwargs['dim'], codebook_sizes=kwargs['codebook_size'], num_quantizers=kwargs['num_quantizers'])
            self.total_codes = np.prod(kwargs['codebook_size'])
            self.indices_to_codes = self.quantizer.get_output_from_indices
        else:
            raise ValueError('quantizer_type must be one of "fsq", "residual_fsq", "vq", "residual_vq"')
    
    def forward(self, x):
        out = self.quantizer(x)
        if len(out) == 2:
            out = out + (torch.tensor([0.0]).to(x.device),)
        return out

def apply_layer_norm(x, layer_norm):
    x = x.transpose(1,2)
    x = layer_norm(x)
    x = x.transpose(1,2)
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.elu = nn.ELU()
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = apply_layer_norm(out, self.ln1)
        out = self.elu(out)
        out = self.conv2(out)
        out = apply_layer_norm(out, self.ln2)
        out = self.elu(out)
        return out + x

class Encoder(nn.Module):
    def __init__(self, in_channels, channels, codebook_dim, kernels=[5,3], strides=[2,2,2,3], bias=True):
        super(Encoder, self).__init__()
        padding = (kernels[0]-1)//2
        final_channels = channels*2**(len(strides))
        self.conv_in = nn.Conv1d(in_channels, channels, kernel_size=kernels[0], stride=1, padding=padding, bias=bias)
        self.lstm = nn.LSTM(input_size=final_channels, hidden_size=final_channels, num_layers=2, batch_first=True)
        self.lstm.flatten_parameters()
        self.conv_out = nn.Conv1d(final_channels, codebook_dim, kernel_size=kernels[0], stride=1, padding=padding, bias=bias)
        self.elu = nn.ELU()
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(codebook_dim)
        self.res_blocks = nn.ModuleList()
        for i in range(len(strides)):
            in_channels = channels
            channels = channels*2
            kernel_size = kernels[1]
            self.res_blocks.append(ResidualBlock(in_channels, in_channels, kernel_size, bias=bias))
            self.res_blocks.append(nn.Conv1d(in_channels, channels, kernel_size=3, stride=strides[i], padding=1, bias=bias))
            self.res_blocks.append(nn.LayerNorm(channels))
            self.res_blocks.append(nn.ELU())
    
    def forward(self, x):
        x = x.transpose(1,2)
        out = self.conv_in(x)
        out = apply_layer_norm(out, self.ln1)
        out = self.elu(out)
        for layer in self.res_blocks:
            if isinstance(layer, nn.LayerNorm):
                out = apply_layer_norm(out, layer)
            else:
                out = layer(out)
        out = out.transpose(1,2)
        out, _ = self.lstm(out)
        out = out.transpose(1,2)
        out = self.conv_out(out)
        out = apply_layer_norm(out, self.ln2)
        out = self.elu(out)
        out = out.transpose(1,2)
        return out

class Decoder(nn.Module):
    def __init__(self, out_channels, channels, codebook_dim, kernels=[5,3], strides=[2,2,2,3], bias=True):
        super(Decoder, self).__init__()
        padding = (kernels[0]-1)//2
        final_channels = channels*2**(len(strides))
        self.conv_in = nn.Conv1d(codebook_dim, final_channels, kernel_size=kernels[0], stride=1, padding=padding, bias=bias)
        self.lstm = nn.LSTM(input_size=final_channels, hidden_size=final_channels, num_layers=2, batch_first=True)
        self.lstm.flatten_parameters()
        self.conv_out = nn.Conv1d(channels, out_channels, kernel_size=kernels[0], stride=1, padding=padding, bias=bias)
        self.elu = nn.ELU()
        self.ln1 = nn.LayerNorm(final_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.res_blocks = nn.ModuleList()
        for i in range(len(strides)):
            in_channels = final_channels
            final_channels = final_channels//2
            kernel_size = kernels[1]
            self.res_blocks.append(nn.ConvTranspose1d(in_channels, final_channels, kernel_size=3, stride=strides[::-1][i], padding=1, output_padding=strides[::-1][i]-1, bias=bias))
            self.res_blocks.append(nn.LayerNorm(final_channels))
            self.res_blocks.append(nn.ELU())
            self.res_blocks.append(ResidualBlock(final_channels, final_channels, kernel_size, bias=bias))
    
    def forward(self, x):
        x = x.transpose(1,2)
        out = self.conv_in(x)
        out = apply_layer_norm(out, self.ln1)
        out = self.elu(out)
        out = out.transpose(1,2)
        out, _ = self.lstm(out)
        out = out.transpose(1,2)
        for layer in self.res_blocks:
            if isinstance(layer, nn.LayerNorm):
                out = apply_layer_norm(out, layer)
            else:
                out = layer(out)
        out = self.conv_out(out)
        # out = apply_layer_norm(out, self.ln2)
        out = out.transpose(1,2)
        out = torch.tanh(out)
        return out


if __name__ == '__main__':
    # Test Quantizer
    # cfg_dict = {'quantizer_type':'vq', 'dim': 256, 'codebook_size': 1024}
    # quant = Quantizer(**cfg_dict)
    x = torch.randn(1,8,256)
    print('next VQ')
    quantizer = Quantizer(quantizer_type='vq', dim=256, codebook_size=1024).eval()
    out = quantizer(x)
    for element in out:
        print(element.shape)
    print('num codes:', quantizer.total_codes)
    print('codes', quantizer.indices_to_codes(out[1]).shape)
    assert torch.all(out[0] == quantizer.indices_to_codes(out[1]))
    # out[-1].sum().backward()
    print('next residualVQ')
    quantizer = Quantizer(quantizer_type='residual_vq', dim=256, codebook_size=[8,8,16], num_quantizers=3).eval()
    out = quantizer(x)
    for element in out:
        print(element.shape)
    print('num codes:', quantizer.total_codes)
    print('codes', quantizer.indices_to_codes(out[1]).shape)
    assert torch.all(out[0] == quantizer.indices_to_codes(out[1]))
    # out[-1].sum().backward() 
    print('next FSQ')
    quantizer = Quantizer(quantizer_type='fsq', dim=256, levels=[8,5,5,5])
    out = quantizer(x)
    for element in out:
        print(element.shape) if element is not None else print('None')
    print('num codes:', quantizer.total_codes)
    print('codes', quantizer.indices_to_codes(out[1]).shape)
    assert torch.all(out[0] == quantizer.indices_to_codes(out[1]))
    # out[0].sum().backward()
    print('next residualFSQ')
    quantizer = Quantizer(quantizer_type='residual_fsq', dim=256, levels=[[3,3],[3,3],[5,3]], num_quantizers=3)
    out = quantizer(x)
    for element in out:
        print(element.shape) if element is not None else print('None')
    print('num codes:', quantizer.total_codes)
    print('codes', quantizer.indices_to_codes(out[1]).shape)
    assert torch.all(out[0] == quantizer.indices_to_codes(out[1]))
    # test backward pass
    # out[0].sum().backward()
    # in_channels = 7
    # channels = 16
    # out_channels = 7
    # codebook_dim = 256
    # kernels = [5,3]
    # strides = [2,2,2,2]
    # encoder = Encoder(in_channels, channels, codebook_dim, kernels, strides)
    # x = torch.randn(1,7,96)
    # out = encoder(x)
    # print(out.shape)
    # decoder = Decoder(out_channels, channels, codebook_dim, kernels, strides)
    # out = decoder(out)
    # print(out.shape)
    # print('number of parameters:', sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad))

        