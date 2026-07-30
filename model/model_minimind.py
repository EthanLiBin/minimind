import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 每个token的向量维度，贯穿整个模型的“通道宽度”。768是小模型的典型值
        self.hidden_size = hidden_size
        # Transformer的层数。每层 = Attention + FFN（或者MoE）,8层是比较浅的配置
        self.num_hidden_layers = num_hidden_layers
        # 是否把FFN换成MoE
        self.use_moe = use_moe
        # 注意力权重和残差连接的dropout概率，0就是不做dropout
        self.dropout = kwargs.get("dropout", 0.0)
        # 词表大小，6400比较小
        self.vocab_size = kwargs.get("vocab_size", 6400)
        # bos(开始)/eos(结束)token的id，生成时遇到eos_token_id就停
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # 是否优先使用pyTorch内置的F.scaled_dot_product_attention（Flash Attention）。True 时在支持的场景下用融合算子，速度和显存都更优。
        self.flash_attn = kwargs.get("flash_attn", True)
        # Q的head维度。8个头 * 96维/头 = 768(hidden_size)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        # K/V的head数量。4 < 8，代表GQA：每2个Q head共享1对KV head，用这种方式来省KV cache
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # 每个head的维度，满足hidden_size/num_attention_heads
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # FFN的激活函数，默认时SiLU -> x * sigmoid(x)。小于0的数置为0
        # 0和1之间平滑过渡，实现线性弯折，满足更复杂场景
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # FFN 中间层的维度（升维后的宽度）
        # math.pi -> 3.14159。比传统的 4 × hidden_size = 3072 小，是 Llama 3 的做法，把 FFN 宽度从 8/3 倍改成 ≈π 倍，更省参数。ceil(... / 64) × 64 是为了对齐到 64 的倍数，方便 GPU 计算。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # 模型支持的最大序列长度(32K)，RoPE 会预计算这个长度的 cos/sin 表。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # RMSNorm的epsilon，防止除零。1e-6是常规值
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # RoPE 的基频参数。公式是 freq = 1 / (theta^(2i/d))。1e6 是 Llama 3 的值，比原始的 10000 更大，高频分量衰减更慢，天然支持更长的上下文。
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # 输入embedding和输出lm_head是否共享权重？True时lm_head.weight = embed_tokens.weight，省vocab_size * hidden_size个参数
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        
        # 长度外推 YaRN相关
        # 推理时是否启用 YaRN 外推，让训练时只见过短文本（2048）的模型能处理长文本（32K）
        """核心逻辑（在 precompute_freqs_cos_sin 里）：

            频率 i 在 [beta_slow, beta_fast] 之间时，做线性插值：
            freq_new = freq_old × (1 - ramp + ramp / factor)

            低频（ramp=1）→ 除以 16，等效于把位置缩小 16 倍
            高频（ramp=0）→ 不变，短距离的位置关系不受影响
            中间的频率 → 线性过渡"""
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,    # 高频截断阈值，控制哪些频率保持不变
            "beta_slow": 1,     # 低频截断阈值，控制哪些频率被完全缩放
            "factor": 16,       # 缩放因子：32K / 2048 ≈ 16
            "original_max_position_embeddings": 2048,    # 训练时的最大长度
            "attention_factor": 1.0,     # 注意力熵的修正系数
            "type": "yarn"      # 外推方法类型
        } if self.inference_rope_scaling else None
        
        ### MoE相关
        # 专家总数，MoE用多个FFN专家替代单一FFN
        self.num_experts = kwargs.get("num_experts", 4)
        # 每个token激活几个专家？1=每个token只走1个专家，计算量和普通FFN一样但模型总容量更大
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        # 每个专家的FFN中间层维度，默认和普通FFN一样大
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # 是否对top-k选中的专家概率做归一化？True 时选中的概率除以它们的和，保证概率和为 1。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        # 负载均衡辅助损失的系数。防止所有 token 都选同一个专家。越大负载越均衡，但可能影响模型质量。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

# RMSNorm = 把每个 token 的 hidden state 向量除以其 RMS（均方根），让每层输出的"尺度"始终稳定在 1 附近。不要"居中"只要"缩放"，既快又好。
# * RMSNorm后，每个 token 的 768 个维度整体 RMS ≈ 1，所有维度在同一个量级上平等对话，Attention 才能真正从"哪个维度更重要"的角度去加权。
# 语义上就是把同一个 token 内部不同特征维度的值拉到同一个数量级(统一尺度)，让后面 Attention 的点积不会某个维度独大。
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # -1 就是 hidden_size 那个维度。比如 hidden_size=768，那每一行的 768 个数内部自己做归一化，不同行互不影响。
        # x.pow(2) x的平方
        # mean(-1, keepdim=True) 沿着hidden_size的维度做平均，维度保持不变
        # eps 防止除以0
        # rsqrt -> 取倒数平方根，也就是1/平方根 -> x * torch.rsqrt -> x/平方根
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 首先算出每对维度的基础频率
    # 频率从 1 一直衰减到约 0.001——相差近 1000 倍。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # 每个位置的角度 = 位置 × 频率
    # 高频维度对位置敏感（差几个 token 就能区分），低频维度对长距离保持感知。
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 取 cos 和 sin
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        
        self.n_local_heads = config.num_attention_heads
        # kv cache的head数量
        self.n_local_kv_heads = self.num_key_value_heads
        # n_rep 代表GQA：每2个Q head共享1对KV head，用这种方式来省KV cache
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度 hidden_size / num_attention_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        
        # nn.Linear(d_in, d_out) = 把最后一维 d_in 个值，通过线性组合 + 偏置，变成 d_out 个值。
        # 定义Q、K、V
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        # 输出线性层
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # 归一化
        # 标准 Attention 没有 QK Norm，容易因为某些 head 的 Q/K 范数过大或过小导致训练不稳定。对每个 head 做 RMSNorm 强制把所有 head 拉到同一尺度，训练更稳，可以用更大的学习率*。代价几乎为零（每个 token 每层多两次小归一化）。
        
        # RMSNorm 之后向量 RMS ≈ 1，再做 RoPE 旋转，旋转不会改变向量的长度（旋转是等距变换），所以加了 RoPE 之后量级依然稳定。如果反过来先 RoPE 再 Norm，逻辑上也说得通，但先 Norm 再 RoPE 更常见。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x.shape是(batch_size, seq_len(序列长度), hidden_size)
        bsz, seq_len, _ = x.shape
        # 应用nn.Linear，x添加向量权重
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # view转置 -> 切分dim=-1维度，通过转换来实现计算
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 使用RMSNorm对Q和K做归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        
        # RoPE：加在"每一层的 Q/K 上"
        # Input Token → Embedding → 进入 Transformer
        
        # 位置编码不在入口，而是每一层都重新施加。因为：
        # RoPE 不改变向量的长度，只改变方向（旋转）。 一层 Attention + FFN 之后，向量的方向被改变了，位置信息跟着就乱了。如果只在入口做一次 RoPE，传到后面几层位置信息就没了。每一层都在 Q/K 上重新旋转，保证每一层的注意力都能拿到干净的位置信号。
        # 为什么只旋 Q 和 K，不旋 V？因为位置信息只需要影响"谁关注谁"：
        # Q @ K^T 决定了 token i 对 token j 的关注程度 → 这里需要知道相对位置；V 是被关注的对象，V 不带位置信息，关注权重拿到后直接加权求和
        # 这是 RoPE 设计的精妙之处：位置信息只改变注意力权重，不污染语义内容。V 里存的是"这个 token 说了什么"，Q 和 K 里除了语义还带上了"我在哪里 / 谁该看我"。
        cos, sin = position_embeddings
        # 直观理解：在 96 维空间里，把 Q 和 K 按照它们在序列中的位置做一个旋转。位置 i 的 token 旋转角度是 i × freq。两个 token 的点积只取决于它们的相对位置差，而不是绝对位置。
        
        """
            position_embeddings 是预计算好的 cos/sin 表，形状都是 [seq_len, head_dim]。
            当前 xq 形状：[bsz, seq_len, n_heads, head_dim]，xk 形状：[bsz, seq_len, n_kv_heads, head_dim]。
            apply_rotary_pos_emb 做的事：
                
            def rotate_half(x): 
                # 把向量对半切，前一半取负拼到后一半后面
                return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
        """
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 拼接历史 KV cache:只在推理时触发（逐 token 生成时）。
        # 简单说就是把新 token 的 K/V 追加到历史的后面，这样计算注意力时能看到全部上下文。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 决定是否返回本次的 KV cache
        past_kv = (xk, xv) if use_cache else None
        # 转置 + 复制 KV head
        # repeat_kv(xk, self.n_rep)：n_rep = 8 // 4 = 2，把 KV head 数从 4 复制到 8，跟 Q head 数对齐：
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 走flash attention，还是自己写？
        # seq_len > 1输入不止1个token；not self.is_causal or past_key_value is None -> 要么不 causal，要么是第一次推理；
        # attention_mask is None or torch.all(attention_mask == 1) -> 没传 mask，或者 mask 全是 1
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # PyTorch 内置的融合算子，内部用了 Flash Attention 算法：不显式构造 [seq, seq] 的注意力矩阵，而是分块计算、边算边累加。省显存，速度快。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # (Q @ K) / head_dim的平方根
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #  Causal Mask（自回归约束）
            # 注意 [:, :, :, -seq_len:] 是只在当前输入的 seq_len 范围内做 mask。这是 KV cache 场景：scores 矩阵的 key 维度包含历史 token，不能对历史 token 上三角 mask（token 0 不能看 token 1，但可以看到历史 token 100）。
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 自定义 attention mask
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax + dropout + 加权求和
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 切换回(bsz, seq_len, hidden_size)
        # -1 -> 自动推导这个维度应该是多少
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# SwiGLU FFN 门控激活函数
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        # FFN 中间层的维度（升维后的宽度）
        intermediate_size = intermediate_size or config.intermediate_size
        # 产生门控信号
        # 它对输入 x 做线性变换后经过 SiLU 激活，得到一个 (0, ~∞) 的 gating 值，然后与 up_proj(x) 逐元素相乘。效果是每个隐藏维度都可以被"关小"或"开大"，让网络自适应地控制信息流——这是 GLU（Gated Linear Unit）的核心思想。
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # FFN的激活函数，默认时SiLU -> x * sigmoid(x)。小于0的数置为0
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # gate_output = self.act_fn(self.gate_proj(x))    # [bsz, seq, 2432]，每个维度是一个"阀门"
        # up_output   = up_proj(x)            # [bsz, seq, 2432]，每个维度是一个"信号"
        # down_proj -> 转回hidden_size维度
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# 路由层面的门控
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 这个 gate 做的是路由选择：决定每个 token 分派给哪几个专家，输出的是 token-to-expert 的分配权重。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # config.num_experts:专家总数，MoE用多个FFN专家替代单一FFN
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    # 用多个小 FFN（专家）替代一个大 FFN，每个 token 只激活其中少数几个
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 3D转2D，做拍平处理。-1 让 PyTorch 自动推导第一维的大小，等于 batch_size × seq_len
        # 第二维保持 hidden_dim 不变
        # x_flat -> (batch_size × seq_len, hidden_dim)
        x_flat = x.view(-1, hidden_dim)
        # self.gate(x_flat) -> 输出该token对每个专家的得分
        # 路由机制：gate 线性层输出每个 token 对每个专家的得分，通过 softmax + top-k 选出每 token 激活的 num_experts_per_tok 个专家
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 对scores做top-k处理，topk_weight：最高的2个分数；topk_idx：最高分的专家编号
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # (可选)归一化处理，值/总和，和为1
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        # 循环的作用：把每个 token 交给它选中的 k 个专家分别算，加权求和，写回对应位置。
        # zeros_like 全是0，形状是x_flat
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):   # 遍历每个专家
            mask = (topk_idx == i)  # 找出哪些 token 被分配给了专家 i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()    # 被分配到专家 i 的 token 编号
                weight = topk_weight[mask].view(-1, 1)  # 对应的权重，形状 [n, 1]
                # index_add_ 做的是按索引累加
                # 循环走完所有专家后，y 里就是每个 token 由它命中的专家输出加权累加的结果。
                # (expert(x_flat[token_idx]) -> 专家xx的输出 * weight(乘上各自的权重)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 幽灵梯度（训练技巧）
                # 那些没有被任何 token 选中的专家，正常 forward 不会参与计算，就不会有梯度。但优化器可能因为它们没梯度而报错，或者我们想让所有专家参数都在计算图中。这行代码的效果是：0 × 参数和 = 0，数值上不影响结果，但把所有专家参数都纳入了计算图，保证每个专家都能收到梯度。非训练时不需要。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 负载均衡损失的系数
            # 防止 router 偷懒——把大部分 token 都分配给一两个专家：
            # 乘以 router_aux_loss_coef 控制惩罚力度，乘以 num_experts 做缩放。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        # 恢复形状
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 注意力机制
        self.self_attn = Attention(config)
        # RMSNorm -> x / (x平方的平均值)的平方根，用来统一hidden_size(768维)的尺度
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN 激活函数
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    # 前向传播
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # hidden_states 就是从模型底层一直往上流动的token表示张量

        # 它就像一条流水线：每个 token 进入时只是一个词 ID，经过嵌入变成 768 维向量，然后每一层 block 都在这个向量上施加注意力交互（token 之间互通信息）和 FFN 变换（每个 token 独立深化理解），最终输出时这 768 维里"装满"了该 token 的语义信息，可以直接投影到词表预测下一个 token。
        residual = hidden_states
        # 输入 hidden_states
            # │
            # ├──→ RMSNorm → Attention → + (残差) ──→ hidden_states'
            # │
            # └──→ RMSNorm → FFN/MOE  → + (残差) ──→ 输出
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词表embed_tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # dropout 一种正则化技术，用于防止模型过拟合。在训练时，它会以 drop_rate 的概率，随机地将输入元素的一部分置为零。这能迫使网络学习到更鲁棒的特征。在模型评估（推理）时，它会自动失效。
        self.dropout = nn.Dropout(config.dropout)
        # num_hidden_layers定义了多层(模型深度)
        # MiniMindBlock -> Transformer模型块，layers -> Transformer块组成的层
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 这里一次性算出了所有位置、所有维度的旋转角度。
        # 为什么要算 cos/sin？
        """
        RoPE 本质上就是把一个向量在 2D 子空间里旋转。旋转一个 2D 向量 (x, y) 角度 θ：
            x' = x·cos(θ) - y·sin(θ)
            y' = x·sin(θ) + y·cos(θ)
            那对于一个 head_dim = 96 的向量，怎么旋转？拆成 48 对，每一对在自己的 2D 平面里旋转：

            维度对 0:  (dim_0, dim_1)   旋转角度 θ_0
            维度对 1:  (dim_2, dim_3)   旋转角度 θ_1
            维度对 2:  (dim_4, dim_5)   旋转角度 θ_2
            ...
            维度对 47: (dim_94, dim_95) 旋转角度 θ_47
            每对的旋转角度不一样：高频维度转得快，低频维度转得慢。
        """
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # 整体流程：把 token ID 变成向量 → 算出这些 token 在序列中的绝对位置 → 把位置信息以 cos/sin 的形式传给每一层 Attention，让它在做旋转位置编码时使用。
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        
        batch_size, seq_length = input_ids.shape
        # 处理 HuggingFace 传来的 KV cache
        # HuggingFace 的 DynamicCache 对象有 .layers 属性，但这份代码用的是自己的一套 KV cache 格式 [None, (k,v), (k,v), ...]。如果外部传进来的是 HF 格式，直接丢弃当没有缓存处理——兼容性保护。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 初始化缺失的 KV cache
        # 如果 past_key_values 是 None（首次推理），初始化为全是 None 的列表，长度等于层数。第一轮 prefill 时每层的 cache 都是 None，后续每轮逐步填充。
        past_key_values = past_key_values or [None] * len(self.layers)
        # 确定 RoPE 的起始位置
        # KV cache 存的是 (key, value)，key 的形状是 [batch, seq_len, ...]，所以 .shape[1] 就是已缓存的 token 数，也就是新 token 在序列中的起始位置
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # 词表嵌入 + Dropout，拿到输入
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # RoPE 频率修复（transformers ≥5.x 的坑）
        # 新版本 transformers 的 meta device 初始化会把 buffer 置零。[0, 0] == 0 就是检测这个情况——如果发现全零，当场重算一遍 RoPE 的 cos/sin 表。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        # 切片出当前步需要的旋转位置编码
        # freqs_cos 和 freqs_sin 是预计算好的全量位置编码表（预训练时算到了 max_position_embeddings 个位置）。这里只切出当前 token 对应的那一段
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)

        # 每一层 MoE 都有自己独立的路由器，各自独立地做 token-to-expert 分配。如果第 3 层负载很均衡但第 7 层所有 token 都涌向同一个专家，第 7 层就浪费了其他专家的容量。
        # 把各层 aux_loss 求和，相当于告诉优化器：所有层的路由器都要各自均衡，哪一层不均衡就惩罚哪一层。
        # 这和标准 MoE 论文（Switch Transformer、GShard）的做法一致——总负载均衡损失 = 所有层负载均衡损失之和。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# PreTrainedModel：它就是一个基础设施层——不参与前向传播（forward 是你自己写的），但管好了一切"模型运行时"的周边事务。
# PreTrainedModel：模型序列化、权重管理等
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # 定义模型
        self.model = MiniMindModel(self.config)
        # 定义模型头
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 给model传参数
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        
        # 只计算需要的 token 的 logits，省显存
        # 其中 slice 是 Python 内置的惰性切片对象（不是 numpy 的），等价于 hidden_states[:, -logits_to_keep:, :]，但更高效。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 应用输出层
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # logits是模型预测的结果，labels是真实结果，比较差距，ignore_index=-100 表示 padding 位置不算
        if labels is not None:
            # logits[..., :-1, :]去掉最后一个位置的预测（没标签了）；labels[..., 1:]：去掉第一个位置的标签（没有预测对应它）
            # logits[0] 预测 → labels[1]("爱")       位置0输出 → 对应下一个位置的真值
            # logits[1] 预测 → labels[2]("中国")     位置1输出 → 对应下下个位置的真值
            # logits[2] 预测 → labels[3]("的")       位置2输出 → ...
            # logits: [batch, seq, vocab_size]
            # labels: [batch, seq]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # cross_entropy = softmax → 取正确项(label)的概率 → 取负对数。概率越接近 1，loss 越接近 0；概率越接近 0，loss 趋向无穷。梯度下降就朝着"让正确项概率变大"的方向优化。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        # 用MoeCausalLMOutputWithPast定义一个dataclass
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    
    # 输入: "我是谁"
    # │
    # ▼
    # ┌─────────────────────────────┐
    # │  第1轮: forward("我是谁")    │  prefill，缓存 KV
    # │  logits → 采样 → "我"       │
    # └─────────────────────────────┘
    # │ past_len 从 0 变成 3
    # ▼
    # ┌─────────────────────────────┐
    # │  第2轮: forward("我")       │  decode，只算新 token
    # │  logits → 采样 → "是"       │
    # └─────────────────────────────┘
    # ▼
    # ┌─────────────────────────────┐
    # │  第3轮: forward("是")       │
    # │  logits → 采样 → "谁"       │
    # └─────────────────────────────┘
    # ▼
    # ┌─────────────────────────────┐
    # │  第4轮: forward("谁")       │
    # │  logits → 采样 → <eos>      │ → finished=true, 停止
    # └─────────────────────────────┘
    # ▼
    # 输出: "我是谁是"
    # 核心逻辑：模型只看上一个新 token，旧信息都在 KV cache 里，一 token 一 token 地往外"吐"，直到够了或遇到 eos。
    
    
    # 自回归文本生成
    # 这个方法就是逐 token 生成文本的完整过程。分解来看：
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 初始化操作
        # 把输入复制 num_return_sequences 份（比如一次生成 3 条结果）
        # finished 跟踪哪些已经结束了（遇到 eos）
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # past_key_values 是变量名，KV cache 是概念名。如果你在别的代码或论文里看到 kv_cache、key_value_states、past_key_values，指的都是它——注意力机制里缓存的 Key 和 Value。
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        if streamer: streamer.put(input_ids.cpu())
        # 逐token循环
        for _ in range(max_new_tokens):
            
            # 首轮：past_len=0，传入全部 input_ids（prefill）
            # 后续轮：past_len 是已有长度，只传新 token，旧的靠 KV cache 回忆
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            
            # temperature 温度缩放
            # 温度控制随机性：<1 更确定（更保守），>1 更随机（更有创意）。
            logits = outputs.logits[:, -1, :] / temperature
            
            # 重复惩罚：已经被生成过的词降低概率，避免车轱辘话
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # 已被生成的词
                    # score / repetition_penalty -> 降正分
                    # score * repetition_penalty -> 升负分（更负）
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                    
            # Top-K: 只保留概率最高的 K 个候选，其余置 -∞（概率变为0）
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P (nucleus sampling): 按概率从高到低累加，加到 p 后截断
            # 这个 mask 是在排序后数组上的——[F, F, T, T, T] 意味着 sorted 索引 [4, 0, 2, 3, 1] 中，保留前 2 个（词4和词0），截断后 3 个（词2、词3、词1）。
            # 但我们需要在原始 logits 上操作。sorted_indices 的作用就是把 mask 映射回去：

            # softmax后概率: [0.5, 0.3, 0.15, 0.04, 0.01]
            # 累加:          [0.5, 0.8, 0.95, 0.99, 1.00]
            # top_p=0.95 → 保留前3个，后2个置-inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # torch.softmax 作用：概率高的更容易抽到，但不是100%
            # argmax（do_sample=False）：永远取概率最高的 → 确定性，可能重复
            # multinomial（do_sample=True）：按概率随机抽 → 有变化，更自然
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            
            # 已标记为 finished 的序列，强行输出 eos 而不继续生成
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            
            # 拼到已有序列后面
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            
            # 检测到 eos 则标记 finished
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                # 全结束了就停
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids