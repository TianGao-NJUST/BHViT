import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        # indicate_small = (input < -1).float()
        # indicate_big = (input > 1).float()
        indicate_leftmid = ((input >= -1.0) & (input <= 0)).float()
        indicate_rightmid = ((input > 0) & (input <= 1.0)).float()

        grad_input = (indicate_leftmid * (2 + 2*input) + indicate_rightmid * (2 - 2*input)) * grad_output.clone()
        return grad_input


class BiTBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val):
        ctx.save_for_backward(input, clip_val)
        out = torch.round(input / clip_val).clamp(0.0, 1.0) * clip_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        q_w = input / clip_val
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_clip_val = ((indicate_middle * (q_w.round() - q_w) + indicate_big) * grad_output).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output.clone()
        return grad_input, grad_clip_val


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs, bias=False, config=None):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            if self.recu:
                #print(scaling_factor, flush=True)
                
                real_weights= real_weights/(torch.sqrt(real_weights.var(dim=-1, keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2-2*0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
                #print(binary_weights, flush=True)
            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)
        
        out = nn.functional.linear(input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

class QuantizeConv2d2(nn.Conv2d):
    def __init__(self,  *kargs, bias=True, config=None):
        super(QuantizeConv2d2, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input,recu=False):
        if self.weight_bits == 1:
            # This forward pass is meant for only binary weights and activations
            real_weights = self.weight
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            real_weights = real_weights - real_weights.mean([1,2,3], keepdim=True)
            
            
            if self.recu:
                
                real_weights = real_weights / (torch.sqrt(real_weights.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                #print(scaling_factor, flush=True)
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2-2*0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
                #print(binary_weights, flush=True)
            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)
        
        out = nn.functional.conv2d(input, weight, stride=self.stride)
        
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out
class QuantizeConv2d(nn.Conv2d):
    def __init__(self,  *kargs, bias=True, config=None):
        super(QuantizeConv2d, self).__init__(*kargs, bias=bias)
        self.weight_bits = config.weight_bits
        self.input_bits = config.input_bits
        self.recu = config.recu
        if self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        elif self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.weight_bits < 32:
            self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
        if self.input_bits == 1:
            self.act_quantizer = BinaryQuantizer
        elif self.input_bits == 2:
            self.act_quantizer = TwnQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        elif self.input_bits < 32:
            self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
 

    def forward(self, input,recu=False):
        if self.weight_bits == 1:
            # This forward pass is meant for only binary weights and activations
            real_weights = self.weight
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            real_weights = real_weights - real_weights.mean([1,2,3], keepdim=True)
            
            if recu:
                #print(scaling_factor, flush=True)
                real_weights = real_weights / (torch.sqrt(real_weights.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
                EW = torch.mean(torch.abs(real_weights))
                Q_tau = (- EW * np.log(2-2*0.92)).detach().cpu().item()
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -Q_tau, Q_tau)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
                #print(binary_weights, flush=True)
            else:
                scaling_factor = scaling_factor.detach()
                binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
                cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
                weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        elif self.weight_bits < 32:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight

        if self.input_bits == 1:
            input = self.act_quantizer.apply(input)
        
        out = nn.functional.conv2d(input, weight, stride=self.stride, padding=self.padding,dilation=self.dilation,groups=self.groups)
        
        if not self.bias is None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs,padding_idx=None, config=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
class ElasticQuantAttention(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input,alpha,alpha2,alpha3, num_bits):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        Qn = 0
        Qp = 1
        eps = torch.tensor(0.00001).float().to(alpha.device)  
        q_w_b = torch.round(input).clamp(0.0, 1.0)       
        T2_mask=(input> 0.9*input.max(dim=3,keepdim = True)[0]).float()
        T1_mask=(input> 0.7*input.max(dim=3,keepdim = True)[0]).float()
        T0_mask=q_w_b-T1_mask
        T1_mask=T1_mask-T2_mask
        alpha = torch.where(alpha > eps, alpha, eps)
        alpha2 = torch.where(alpha2 > eps, alpha2, eps)
        alpha3 = torch.where(alpha3 > eps, alpha3, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        assert alpha2 > 0, 'alpha2 = {:.6f} becomes non-positive'.format(alpha2)
        assert alpha3 > 0, 'alpha3 = {:.6f} becomes non-positive'.format(alpha3)   
        ctx.save_for_backward(alpha,alpha2,alpha3)
        ctx.other = T1_mask,T2_mask,T0_mask
        
        
        w_q = T0_mask* alpha + T1_mask*alpha2 + T2_mask*alpha3  
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None,None, None, None

        alpha,alpha2,alpha3 = ctx.saved_tensors
        T1_mask,T2_mask,T0_mask = ctx.other
        #####
        #####        
        grad_alpha = (T0_mask * grad_output).sum().unsqueeze(dim=0)
        grad_alpha2= (T1_mask * grad_output).sum().unsqueeze(dim=0)
        grad_alpha3= (T2_mask * grad_output).sum().unsqueeze(dim=0)
        
        
        grad_input = (alpha*T0_mask+ T1_mask * alpha2+ T2_mask * alpha3 )* grad_output.clone()
        
        return grad_input,grad_alpha, grad_alpha2,grad_alpha3, None  
class GSB_Attention(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input,alpha,alpha2,alpha3, num_bits):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        Qn = 0
        Qp = 1
        eps = torch.tensor(0.00001).float().to(alpha.device)  
        q_w = input / alpha
        q_w_b = torch.round(q_w).clamp(0.0, 1.0)    
        T2_mask=(q_w> 0.9*q_w.max(dim=3,keepdim = True)[0]).float()
        T1_mask=(q_w> 0.7*q_w.max(dim=3,keepdim = True)[0]).float()
        T0_mask=q_w_b-T1_mask
        T1_mask=T1_mask-T2_mask
        alpha = torch.where(alpha > eps, alpha, eps)
        alpha2 = torch.where(alpha2 > eps, alpha2, eps)
        alpha3 = torch.where(alpha3 > eps, alpha3, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        assert alpha2 > 0, 'alpha2 = {:.6f} becomes non-positive'.format(alpha2)
        assert alpha3 > 0, 'alpha3 = {:.6f} becomes non-positive'.format(alpha3)   
        ctx.save_for_backward(input,alpha,alpha2,alpha3)
        ctx.other = T1_mask,T2_mask,T0_mask
        
        
        w_q = (T0_mask + T1_mask*alpha2 + T2_mask*alpha3)* alpha  
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None,None, None, None

        input,alpha,alpha2,alpha3 = ctx.saved_tensors
        T1_mask,T2_mask,T0_mask = ctx.other
        #####
        q_w = input / alpha
        indicate_small = (q_w < 0.0).float()
        indicate_big = (q_w > 1.0).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        ##### 
        h = indicate_middle*(-q_w)
        g1=  T0_mask*(1.0+h)
        g2= alpha2*T1_mask*(1.0+h)
        g3= alpha3*T2_mask*(1.0+h)
        grad_alpha = ((g1+g2+g3) * grad_output).sum().unsqueeze(dim=0)
        grad_alpha2= (alpha*T1_mask * grad_output).sum().unsqueeze(dim=0)
        grad_alpha3= (alpha*T2_mask * grad_output).sum().unsqueeze(dim=0)
        
        
        grad_input = alpha*(T0_mask + T1_mask * alpha2+T2_mask * alpha3)* grad_output.clone()
        
        return grad_input,grad_alpha, grad_alpha2,grad_alpha3, None     


class BinaryActivation_Attention(nn.Module):
    def __init__(self, num_head, nbits_a=4, **kwargs):
        super(BinaryActivation_Attention, self).__init__()
        self.num_head = num_head
        self.alpha = nn.Parameter(torch.ones([num_head]))
        self.zero_point = nn.Parameter(torch.zeros([num_head]))
        self.register_buffer('init_state', torch.zeros(1))
        self.nbits = nbits_a
        # print(self.alpha.shape, self.zero_point.shape)
    def grad_scale(self,x, scale):
         y = x
         y_grad = x * scale
         return y.detach() - y_grad.detach() + y_grad
    def round_pass(self,x):
         y = x.round()
         y_grad = x
         return y.detach() - y_grad.detach() + y_grad
    def forward(self, x):
        if self.alpha is None:
            return x
        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            
            Qn = 0
            Qp = 2 ** (self.nbits - 1) - 1
            
            self.alpha.data.copy_(2 * x.abs().mean(dim=-1).mean(dim=-1).mean(dim=0)/ math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * x.detach().min(dim=-1)[0].min(dim=-1)[0].min(dim=0)[0] - self.alpha.data * Qn)
            self.init_state.fill_(1)

        
        Qn = 0
        Qp = 2 ** (self.nbits - 1) - 1
        

        g = self.num_head / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = self.grad_scale(self.alpha, g)
        zero_point = self.grad_scale(self.zero_point, g)
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = self.round_pass(x / alpha + zero_point).clamp(Qn, Qp)
        x = x * alpha

        return x




class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2* tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4* tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp
        elif init_method == 'attention':
            init_val =2* (tensor.sum()/tensor.sign().sum() )/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.sum()/tensor.sign().sum() ) / math.sqrt(Qp)
        elif init_method == 'attention1':
            n = tensor[0].sign().clip(0,1).sum()-tensor[1].sign().clip(0,1).sum()
            n1 = ((tensor[0].sign().clip(0,1)-tensor[1].sign().clip(0,1))*tensor[2].round().clip(0,1)).sum()
            init_val = (tensor[0].sum()-tensor[1].sum()-n1)/n
        elif init_method == 'attention2':
            n1 = (tensor[0].sign().clip(0,1)*tensor[1].round().clip(0,1)).sum()
            init_val = ((tensor[0].sum()-n1)/tensor[0].sign().clip(0,1).sum())/ math.sqrt(Qp) 
        else:
            init_val = 2*(tensor.abs().sum()/tensor.sign().abs().sum())/ math.sqrt(Qp) if symmetric \
                else 4*(tensor.abs().sum()/tensor.sign().abs().sum() ) / math.sqrt(Qp)
       
        self._initialize(init_val)