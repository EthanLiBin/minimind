import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values

# 奖励函数
"""最终 reward = 
    ±0.5  (长度好不好)
  + ±0.5~1.25 (思考格式对不对)
  - 0~0.5 (重复惩罚)
  + 奖励模型打分 (语义质量，-∞~+∞)"""
def calculate_rewards(prompts, responses, reward_model):
    # 根据response生成全是0的奖励变量
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        # 把带格式标签的 prompt 拆成结构化的对话列表，以便传给奖励模型

        # 解析 prompt:
        # "<|im_start|>user\n什么是光合作用<|im_end|>\n<|im_start|>assistant\n"
        # 解析后:
        # messages = [{"role": "user", "content": "什么是光合作用"}]
        # answer   = "光合作用是植物利用光能..."    ← 模型生成的回复
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            # 规则1:长度控制。回复<20字符(太短) -> -0.5；回复字符太长 > 800 -> -0.5
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            # 规则2：思考格式
            if '</think>' in response:  # 模型输出中包含了思考
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5   # 思考部分不能太短/太长
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25    # </think> 只能出现一次
                answer = answer_content.strip() # 把思考部分去掉，只留答案给 reward model
            # 规则3:重复惩罚，检测 n-gram 重复率，重复太多扣分
            rewards[i] -= rep_penalty(answer)

            # 奖励模型打分，返回: 比如 3.2，输出一个标量分数
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 合并：最终 reward = 硬规则分数 + 奖励模型分数
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    # 训练策略模型、价值网络模型
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # 转tokenizer
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # rollout环节：根据输入生成回复
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        # gen_out -> 回复 response值
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        
        # 获取奖励模型的分数
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # 准备阶段：确定"哪些 token 参与训练"
        # full_mask：整条序列（prompt + response）里非 pad 的位置。它是给 **critic** 用的（critic 要对整条序列每个位置估 value，pad 位置没意义）。
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        # `labels = gen_out[:, 1:]`：**next-token 移位**。`logits[:, :-1]` 的第 i 个位置预测的是第 i+1 个 token，所以 label 要右移一位才能对上。
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        
        # `logp_pos` 是"取 logp 时该从第几个位置取"的索引
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        # 回复 token 在 gen_out 里的绝对位置
        # 位置对齐，那个 -1 就是移位逻辑，最容易写错
        # 语言模型是"预测下一个 token"的：logits 位置 i 预测的是位置 i+1 的那个 token。所以一个 token 在 gen_out 里的位置是 j，它的 logp 就得去 logits 的位置 j−1 找。那个 -1 就是这么来的。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        
        # 处理 EOS 截断：回复长度到底算到哪
        # **EOS 之后的 token 不该训练**。模型生成"结束符"之后再往下 token 都是 pad 或无关内容，如果也计入 loss 会把模型带偏
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 回复里非 pad 的 token 数
        resp_lengths = resp_pad_mask.sum(dim=1); 
        valid_resp = resp_lengths > 0; 
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1); 
        # 第一个 EOS 的位置
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        # `resp_policy_mask` 就是你**最终训练用的 response 位置掩码**：在 EOS 之前 且 是真实生成的 token。
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        """
        old_resp_values：旧 critic 对每个 response 位置的 value 预测（`values_seq.gather(1, logp_pos)`，gather 就是在全序列 value 里把 response 位置"抠"出来）
        ref_resp_logp：**参考模型**对每个 response token 的 logp。用于 KL 约束——防止 actor 偏离初始模型太远
        old_resp_logp：actor **旧策略**的 logp（这段代码里没算，说明在代码段上方已经算好了）
        logp = log probability（对数概率） = ln(P(token))，就是"模型给某个 token 的概率"再取个自然对数。
        logp 就是"模型对某个 token 的把握程度"的对数表达，数值上 ln(概率)，永远是负的，越大越有信心。
        """
        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            
            # 奖励回填：稀疏奖励 → 每个 token 的奖励
            # RLHF 的经典套路：**整条回复只拿到一个外部奖励**（reward model 打分 / 规则奖励），但 PPO 需要每个 token 位置的奖励信号。做法是把这唯一一个奖励放到**回复的最后一个 token**上，之后靠 GAE 把价值"向后传播"到前面所有 token。`valid_resp` 过滤掉完全空回复的样本（它们没有 last token 可放）。
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            gen_len = old_resp_values.size(1); 
            lastgaelam = torch.zeros(B, device=args.device); 
            advs_rev = []
            
            # GAE：算每个位置的 advantage 和 returns
            # old_resp_values[:, t + 1] -> V(s_{t+1})
            # - **TD residual**：`δ_t = r_t + γ·V(s_{t+1}) − V(s_t)` → 就是 `delta`。
            # - **GAE 递推**：`A_t = δ_t + γ·λ·A_{t+1}` → 就是 `lastgaelam = delta + γ·λ·lastgaelam`。
            for t in reversed(range(gen_len)):
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R] 
            returns = advantages + old_resp_values  # [B, R]

            # **Advantage 归一化**（几乎必做，能大幅稳定训练）
            # 先标准化再乘 mask，mask 外的位置归零。`+1e-8` 防方差为 0 时除以 0。
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # 这整段里真正"影响训练"的只有 mb_size 和 stop_ppo；七个 _sum 是纯日志，两个 unwrapped 是纯工程。 按照你之前定的优先级——算法看 mb_size/stop_ppo 的机制，日志那组认识就行，DDP 解包记住"剥壳"这个套路即可
        mb_size = max(1, min(args.mini_batch_size, B))  #本轮每个 minibatch 有多大
        stop_ppo = False    # 早停开关
        policy_loss_sum = 0.0   # actor 的 PPO 目标损失
        value_loss_sum = 0.0    # critic 的价值预测误差
        kl_sum = 0.0    # 新旧策略的 KL（approx_kl）
        kl_ref_sum = 0.0    # 对 reference 的 KL 惩罚
        clipfrac_sum = 0.0  # 被 clip 的 token 占比
        aux_loss_sum = 0.0  # MoE 专家负载平衡损失
        log_count = 0   # 除法的分母
        # DDP 解包
        # DDP 那层壳主要管梯度同步，直接拿本体调用前向，省掉壳的开销、也避免访问内部属性时踩壳的坑。
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        
        # PPO 更新循环：核心中的核心
        # 结构是标准的**多 epoch × 多 minibatch**：
        for ppo_epoch in range(args.ppo_update_iters):  #同一批 rollout 数据复用多次
            if stop_ppo:
                break
            b_inds = torch.randperm(B, device=args.device)  #每轮打乱
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]
                
                # critic 重新算，参与反传
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    # actor 重新算
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    # `mb_resp_logp` / `mb_resp_values` 是"当前更新中的策略"
                    # 在 autocast 内计算 log_softmax，避免直接对 fp16/bf16 logits
                    # 计算造成额外数值偏差。
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                # PPO 的**重要性采样**：我们在用旧策略采样的数据来更新新策略，所以要乘上"新/旧"比值来修正分布差异。
                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                               f"mean|lr|={_lrv.abs().mean().item():.6e} "
                               f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                               f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                               f"dropout={getattr(lm_config, 'dropout', None)} "
                               f"training={actor_unwrapped.training}")
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                """同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                早停：approx_kl 超阈值就停
                1. `0.5·log_ratio²` 是 PPO 论文里对 KL 的二阶近似（ratio≈1 时成立）。KL 太大说明新旧策略已经偏离太远，再更新就废了，所以要提前停。
                2. **`all_reduce` 同步**：如果某张卡 KL 超了、另一张没超，一张卡 break 出循环而其它卡还在跑，DDP 的通信对不上就会**死锁**。所以提前把 KL 在全部卡上求平均，让所有卡**一致地**决定要不要停。"""
                approx_kl_val = approx_kl.detach().clone()  # 标准近似 KL
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)    # 跨卡求平均
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                """
                ln -> log()
                ratio = exp(ln(P_new) − ln(P_old))
                    = exp(ln(P_new / P_old))
                    = P_new / P_old        ← 因为 exp 和 ln 互为反函数，抵消了"""
                ratio = torch.exp(log_ratio)
                
                # "clip fraction"，被裁剪的 token 占有效 token 的比例。它是一个诊断指标，不进 loss、不影响训练，只进日志（你前面见到的 clipfrac_sum 累加的就是它）。
                # clipfrac = 被 clip 截断梯度的 token 比例，是 PPO 最直观的稳定性指标——高说明步子太大，低说明太保守，理想值大概在 0.05~0.2。纯监控用，不影响训练。
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                
                # kl_ref_penalty = actor 对 reference model 的 KL 散度惩罚，它是 policy loss 里的"安全绳"，防止 actor 在追奖励时偏离初始模型太远（reward hacking / 输出退化）。数学形式比 clipfrac 复杂一点，但拆开不难。
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                """
                *Policy loss**（clipped PPO 目标 + KL 惩罚）：
                policy_loss = E_mask[ max(-A·ratio, -A·clamp(ratio, 1±ε)) ]  +  kl_coef · kl_ref_penalty
                kl_ref_penalty = 用 z − 1 − ln(z)（z = π_ref/π_actor）在有效 token 上求平均，得到 actor 偏离 ref 的非负惩罚，乘上 kl_coef 加进 policy loss，防止模型在 RL 训练中崩成怪话。
                kl_ref_penalty = exp(ref − mb) − (ref − mb) − 1
                `kl_ref_penalty` 是对 reference model 的偏离惩罚"""
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                """value_loss = 0.5 · E_mask[ max((V − returns)², (clamp(V, old±cliprange) − returns)²) ]
                    和 policy 的 clip 思路一样：如果新 value 离旧 value 太远，就用被限制在 `old ± cliprange` 内的那个平方误差，防止 value 更新步子太大。
                    **aux_loss**：MoE（Mixture of Experts）的辅助平衡 loss；非 MoE 时是 0。
                    
                    为什么必须有 value loss（它干嘛的）？

                    回顾你前面学的因果链：advantage 需要 V(s) 当基线，GAE 需要 V(s_{t+1}) 做 bootstrap。如果 critic 的 V 是错的：
                    - advantage 就错 → policy 更新方向就偏；
                    - GAE 自举就错 → 越来越偏。
                    所以 critic 必须每轮都学着逼近真实回报，下一轮才能给出更好的 V → 更好的 advantage → 更好的策略。value loss 就是"教 critic 预测 V"的损失函数。 它的目标 returns 就是你之前学的 returns = advantages + old_resp_values（GAE 给出的 TD 目标）。
                    """
                    # 它的任务是教 critic 学会预测 V（逼近 GAE 的 returns），因为 advantage 和 GAE 都依赖 V 的准确性；torch.max 的裁剪写法是为了防止价值函数一步更新过猛，和 policy 的 clip 是同一套防抖逻辑。
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                """早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                早停时的"假 backward"——这是本段最精巧的地方
                早停并不是直接 `break`，而是**照常 forward + backward，只是把 loss 乘 0**。这样：

                - 梯度为 0 → 等价于不更新，效果和 break 一样。
                - 但**每个 rank 都走完了相同的 forward/backward 流程**，DDP 的通信闭环完整，不会死锁。
                这是处理"分布式早停"的正确姿势，比"某卡 break 某卡继续"安全得多"""
                
                # **aux_loss**：MoE（Mixture of Experts）的辅助平衡 loss；非 MoE 时是 0。
                # PPO 一次更新 = "顺着 advantage 的方向推策略（但用 clip 保证每一步都小），同时把值函数往真实回报校准（也 clip 防抖），并留一点探索余地（熵 / KL 约束）。三件事一个 loss 打包，一次 backward 一起学。
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                # 反向传播，计算梯度
                loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # 梯度累积与优化器
                # 每攒够 `accumulation_steps` 次 backward，才 clip 梯度、step 一次、清梯度。`grad_clip` 用 `clip_grad_norm_`（范数裁剪）防梯度爆炸——PPO 的 actor 和 critic 是**两套独立的优化器**，因为两者 loss 尺度不同、学习率也不同。
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        # 日志 / 保存 / 清理
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            # 主进程统计：`reward`、`kl_ref`、`approx_kl`、`clipfrac`、`critic_loss`、`avg_response_len`，wandb.log + print。
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # actor 权重转 `.half()` 存 `.pth`（推理/部署用 fp16）；再用 `lm_checkpoint` 存**完整状态**（含 critic、两个优化器、两个 scheduler、epoch/step），用于断点续训。注意 `_orig_mod` 是 torch.compile 包的一层壳，要先 `getattr` 剥掉再取 `state_dict`。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # 最后 `del` 一大串张量：长训练里每步都 new 一堆大张量，显式释放避免"用到了下下下轮"，这属于长期训练的经验性写法。
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型、Ref参考模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # 从checkpoint中恢复
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)

    # 价值网络 V 模型，对每一个token进行期望评估(当前t -> T)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # 奖励模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    # 之前的 pretrain 和 SFT 都是有标准答案的——模型看到上文，预测下一个词，和 labels 对比算 loss。
    # 但 PPO 没有标准答案——模型要自己先说出"心里话"，然后靠奖励模型判断说得好不好。Rollout Engine 就是负责"让模型开口说话"的那个环节。
    # PPO 让模型先说再做——先通过 rollout engine 生成回复，再用奖励信号教会模型说得更好。create_rollout_engine 就是创建这个"发言人"的工厂。
    
    # PPO 的一个 step:
    # ┌─────────────────────────────────────────────────┐
    # │  1. rollout_engine.rollout()                    │
    # │     输入: prompt（"请解释量子力学"）              │
    # │     输出: 模型生成的完整回复 + 每个token的log概率  │
    # │     → "量子力学是研究微观粒子..."                  │
    # └─────────────────────────────────────────────────┘
    #                     ↓
    # ┌─────────────────────────────────────────────────┐
    # │  2. reward_model 打分                            │
    # │     输入: prompt + 生成的回复                     │
    # │     输出: 一个奖励分数（比如 4.2）                 │
    # │     → 好回答分高，烂回答分低                       │
    # └─────────────────────────────────────────────────┘
    #                     ↓
    # ┌─────────────────────────────────────────────────┐
    # │  3. 算优势函数 + PPO 更新参数                     │
    # │     ratio = 新logp / 旧logp                      │
    # │     policy_loss = -advantage × clip(ratio)       │
    # │     → 把高分回复的概率拉高，低分的概率压下去        │
    # └─────────────────────────────────────────────────┘
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 创建dataset
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    # 分布式：DistributedSampler 负责分数据，DistributedDataParallel（DDP）负责同步梯度
    # 每张卡虽然拿到的数据不同，但 gradient 通过 AllReduce 通信求平均后，每张卡得到的是完全一样的平均梯度。然后各自 optimizer.step() 更新的幅度一样，所以所有卡上的模型参数始终保持同步。
    # DataParallel 是分数据、AllReduce 同步梯度；模型权重在每个进程上都有全量拷贝，更新后完全一致。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 策略模型和 价值网络 模型的优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # PPO 的更新次数 = 数据batch × epoch × PPO重复次数 × mini_batch切割数 ÷ 梯度累积。调调度器就是告诉它总共要走多远，然后每步自动前进一步。
    """1 个 epoch:
        ├─ 250 个数据 batch
        │   ├─ batch[0]:
        │   │   rollout 一次 (采 32 条回复)
        │   │   切割成 4 个 mini_batch (每份 8 条)
        │   │   每份重复 PPO 更新 3 次
        │   │   → optimizer.step() 执行了 4×3 = 12 次
        │   │   → scheduler 前进了 12 步
        │   │
        │   ├─ batch[1]: 同上，scheduler 又前进 12 步
        │   │ ...
        │   └─ batch[249]: 同上
        │
        └─ epoch 结束时 scheduler 前进了 250×12 = 3000 步
        2 个 epoch → 总共前进 6000 步
    """
    # 计算有多少个batch
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 算每个bacth会被切成几个mini_batch
    # 为什么要切？PPO 更新时用小批量更稳定，但 rollout 时 batch 大一点效率高。
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    """算优化器总共要更新几次
        含义:
        250 个数据batch
        × 2 个 epoch
        × 3 轮 PPO 更新（同一批数据反复用 3 次）
        × 4 个 mini_batch（每个数据batch 拆成 4 份）
        × 1 (没做梯度累积)
        = 优化器总共更新 6000 次
    """
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    
    # 动态学习率 -> 余弦退火调度器
    # 优化器如何调用？actor_scheduler.step()  -> lr 自动前进一步
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # 如果有checkpoint数据，加载checkpoint数据
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 让你的 Python 代码跑得接近手写 CUDA 的速度，代价是第一次调用时停顿编译几十秒
        # 先编译torch.compile，先把整个计算图看一遍，优化后再跑，省掉中间停顿和重复运算
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)

    # 检测是否做了分布式初始化
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()