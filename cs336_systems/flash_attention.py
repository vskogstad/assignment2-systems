import math
import triton
import triton.language as tl
import torch
from einops import einsum, rearrange, reduce

class TritonFlashAttentionAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q_ptr, K_ptr, V_ptr, is_causal: bool = False):
        b, Nq, d = Q_ptr.shape
        scale = 1 / math.sqrt(d)
        
        
        BLOCK_SIZE = 1024

        

        Bq = 16
        Bk = 16
        Tq = Nq // Bq

        n_programs = Tq * b
        grid = (n_programs)
        flash_fwd_kernel[grid](
            Q_ptr, K_ptr, V_ptr,
            O_ptr, L_ptr,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D: tl.constexpr,
            Q_TILE_SIZE: tl.constexpr,
            K_TILE_SIZE: tl.constexpr,)
        """print(f"this is {Q.shape=}, {K.shape=} and {V.shape=}")
        print(torch.max(torch.tensor(((1, 2, 3), (4, 5, 6)))))
        S = einsum(Q, K, "b sq d, b sk d -> b sq sk") / math.sqrt(d)
        print(f"{S.shape = }")
        s_max = torch.max(S, dim=2, keepdim=True).values
        print(f"{s_max.shape = }")
        numerator = torch.exp(S - s_max)
        denominator = torch.sum(numerator, dim=2, keepdim=True)
        print(f"{numerator.shape = } -- {denominator.shape = }")
        P = numerator / denominator
        print(f"{P.shape = }")
        O = einsum(P, V, "b sq sk, b sk d -> b sq d")
        L = torch.log(reduce(torch.exp(S), "b sq sk -> b sq", "sum"))
        print(f"{O.shape = } -- {L.shape = }")"""
        ctx.save_for_backward(L, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        return NotImplementedError

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # TODO: Set up K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr similarly
    # TODO: Implement the flash attention algorithm 

class FlashAttentionAutogradFuncion(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        print(f"this is {Q.shape=}, {K.shape=} and {V.shape=}")
        b, Nq, d = Q.shape
        b, Nk, d = K.shape
        d_fac = 1/math.sqrt(d)

        Bq = 16
        Bk = 16

        Tq = Nq // Bq
        Tk = Nk // Bk

        O_global = torch.zeros_like(Q)  # b, Nq, d
        L_global = torch.zeros((b, Nq))  # b, Nq

        # calculating each batch independently
        Q_b = torch.split(Q, split_size_or_sections=1, dim=0)
        K_b = torch.split(K, 1, dim=0)
        V_b = torch.split(V, 1, dim=0)

        for batch in range(b):
            # tiling Q, K and V matrices
            Q_tiled = torch.split(Q_b[batch], split_size_or_sections=Bq, dim=1)  # torch split splits by "split size"!
            K_tiled = torch.split(K_b[batch], Bk, dim=1)
            V_tiled = torch.split(V_b[batch], Bk, dim=1)

            # implementing flash attention algo
            for i in range(Tq): #Tq

                q_i = Q_tiled[i].squeeze(0)
                # print(f"{q_i.shape = }")
                m_i = torch.ones((Bq,)) * float("-inf")
                l_i = torch.zeros((Bq,))
                o_i = torch.zeros((Bq, d))
                # print(f"{o_i.shape = }")
                for j in range(Tk): #Tk
                    k_j = K_tiled[j].squeeze(0)
                    v_j = V_tiled[j].squeeze(0)

                    # running softmax
                    s_ij = einsum(q_i, k_j, "Bq d, Bk d -> Bq Bk") * d_fac  # compute pre-softmax attention
                    m_j = torch.max(s_ij, dim=-1).values  # max_value in each row of current tile
                    #print(f"{m_j = } -- {s_ij[:,-1] = }")
                    m_j = torch.max(m_i, m_j)  # update max-values for each row.
                    #print(f"{s_ij = } -- {m_j = }")
                    p_ij = torch.exp(s_ij - m_j.unsqueeze(1))
                    #print(f"{p_ij == torch.exp(s_ij - m_j)} ")
                    # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk
                    l_i = torch.exp(m_i - m_j) * l_i + torch.sum(p_ij, dim=-1)
                    #print(f"-----{torch.exp(m_i - m_j) = } \n----{torch.exp(m_i - m_j).shape = } ")
                    # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
                    o_i = torch.diag(torch.exp(m_i - m_j)) @ o_i + p_ij @ v_j
                    #print(f"{l_i = } -- {o_i = }")
                    # running output. diag() as we here have a matrix of values we need to rescale instead of a scalar for l_i
                    # Bq, Bq @ Bq, d -> Bq, d
                    m_i = m_j # update m j-1
                O_i = torch.diag(1 / l_i) @ o_i
                L_i = m_j + torch.log(l_i)  # TODO: Need to understand wtf this is.
                # write to out
                O_global[batch, i * Bq : (i + 1) * Bq, :] = O_i
                L_global[batch, i * Bq : (i + 1) * Bq] = L_i

        ctx.save_for_backward(L_global, O_global)
        return O_global


    @staticmethod
    def backward(ctx, grad_out):
        return NotImplementedError


class AttentionAutogradFuncion(torch.autograd.Function):
    """Basic implementation to get the logic right"""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        b, s, d = Q.shape
        print(f"this is {Q.shape=}, {K.shape=} and {V.shape=}")
        print(torch.max(torch.tensor(((1, 2, 3), (4, 5, 6)))))
        S = einsum(Q, K, "b sq d, b sk d -> b sq sk") / math.sqrt(d)
        print(f"{S.shape = }")
        s_max = torch.max(S, dim=2, keepdim=True).values
        print(f"{s_max.shape = }")
        numerator = torch.exp(S - s_max)
        denominator = torch.sum(numerator, dim=2, keepdim=True)
        print(f"{numerator.shape = } -- {denominator.shape = }")
        P = numerator / denominator
        print(f"{P.shape = }")
        O = einsum(P, V, "b sq sk, b sk d -> b sq d")
        L = torch.log(reduce(torch.exp(S), "b sq sk -> b sq", "sum"))
        print(f"{O.shape = } -- {L.shape = }")
        ctx.save_for_backward(L, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        return NotImplementedError


