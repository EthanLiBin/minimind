import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

"""完整流水线定位

① rollout/生成：每个 prompt 采样 G 条回复        ← 多次生成发生在这里
   ↓ 产出 responses: [B×G] 条回复
② calculate_rewards（你这个函数）：给每条回复打分  ← 你现在在这
   ↓ 产出 rewards: [B×G] 个标量
③ 组内归一化：reshape 成 [B, G]，算 advantage    ← 下一步，没贴出来
④ GRPO 策略更新"""
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        # args.num_generations 就是 G（组大小）。i * num_generations + j 把 [B×G] 展平的回复重新映射回 [B, G] 的网格——每个 prompt i 占一行（一组），组里有 G 个分数。
        # 这组分数正是 GRPO 组内归一化的"原料"
        # 这就是 GRPO 和 PPO 的分水岭：PPO 的 advantage 靠 critic 网络算；GRPO 不训练 critic，全靠这个函数产出的"一组 G 个奖励"做相对比较。
        for i in range(batch_size): # 遍历 prompt
            for j in range(args.num_generations):    # 遍历同一 prompt 的 G 条回复
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 把带格式标签的 prompt 拆成结构化的对话列表，以便传给奖励模型

                    # 解析 prompt:
                    # "<|im_start|>user\n什么是光合作用<|im_end|>\n<|im_start|>assistant\n"
                    # 解析后:
                    # messages = [{"role": "user", "content": "什么是光合作用"}]
                    # answer   = "光合作用是植物利用光能..."    ← 模型生成的回复
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                # 规则1:长度控制。回复<20字符(太短) -> -0.5；回复字符太长 > 800 -> -0.5
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:  # 模型输出中包含了思考
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5    # 思考部分不能太短/太长
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25 # </think> 只能出现一次
                    answer = answer_content.strip() # 把思考部分去掉，只留答案给 reward model
                # 规则3:重复惩罚，检测 n-gram 重复率，重复太多扣分
                rewards[response_idx] -= rep_penalty(answer)

                # 奖励模型打分，返回: 比如 3.2，输出一个标量分数
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 合并：最终 reward = 硬规则分数 + 奖励模型分数
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards

# grpo训练
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        # 使用tokenizer，用于模型输入
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # 如果有最长限制，就按照最长做截取
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # rollout生成：每个prompt采样G条回复
        # 产出 responses: [B×G] 条回复
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # 对每一个prompt生成[B×G] 条回复，来做奖励打分
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        # 使用混合精度来进行计算
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            # **aux_loss**：MoE（Mixture of Experts）的辅助平衡 loss；非 MoE 时是 0。
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # per_token_logps = 当前策略模型，给"这次生成出来的每个 response token"的对数概率，shape [B, R]。它和你 PPO 代码里的 mb_resp_logp 是同一个东西，只是换了个更直白的名字——"per token"（每个 token 一个 logp）。
            # per_token_logps = 当前模型对"这次生成串里每一个 token"的对数概率 [B, R]。它是 GRPO 策略更新的"活数据"：和旧策略的 old_logps 相减得到 ratio，进 clipped loss，然后梯度反向流入 actor 参数。reward 是"整条回复一个分"，它是"每个 token 一个概率"
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        with torch.no_grad():
            # reference模型生成token的对话概率
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # 如果是debug，按debug_interval节奏打印数据
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # args.num_generations 就是 G（组大小）
        # grouped_rewards的结构是[B, num_gen]
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 拿到平均值
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 拿到标准差
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 计算优势函数
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        """构造一个 mask，标出"哪些 response 位置是有效训练位置"——从开头到第一个 EOS（含 EOS）为 1，EOS 之后和 padding 为 0。
        EOS 之后的 token 不参与训练——模型已经宣告"生成结束"，后面的内容无意义，训了会把分布带偏。这就是这段代码的全部意义。
        先找出每个样本第一个真实 EOS 的位置（argmax，没 EOS 就用末尾兜底），再构造"位置 ≤ EOS 位置"的 mask，并与 padding 掩码相与，得到"有效训练 token"的掩码。整段 = 你之前 PPO 里的 resp_lengths+resp_policy_mask，只是换成了纯 mask 写法。"""
        
        # 拿到"哪些是真实 token"的掩码
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 找出 EOS 在哪
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        # 给每个样本一个"EOS 位置"的默认值
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        # 有 EOS 的样本，把默认值换成"第一个 EOS"的位置
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 用 eos_idx 造最终 mask
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # 计算ref模型和策略模型之间的KL散度
        # exp(log(ref) - log(P_new))
        # = exp(log(ref / P_new))
        # = ref / P_new
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        """cispo 是一个具体的 RL 损失类型，（裁剪重要性采样策略优化），出自 MiniMax-M1 论文。它是 GRPO 家族的一员，但换了一种"裁剪"的哲学。你这段代码正是它的标准实现，我给你彻底讲透。
        
        为什么要这样？——稀有 token 是关键：
        长推理链里，有些 token 概率极低但至关重要：比如 "However"、"Wait"、"Let me recheck"、"Aha" 这类触发"自我反思/纠错"的词。它们在新旧策略下概率都小，ratio 很容易越界。
            - PPO/GRPO 的 min(...) 会把它们的梯度清零 → 模型学不到这些关键转折点。
            - CISPO 永远保留它们的梯度（只限制幅度）→ 思考信号不丢。
        这就是 CISPO 比 GRPO 在推理任务上更稳、更快的原因。"""
        # CISPO = "把重要性权重裁上限、detach 掉，再乘 A·logπ 做 REINFORCE 式更新"——和 GRPO 比，它不裁目标函数而是裁权重，让低概率但关键的推理 token（如 'Wait'/'However'）永远保有梯度，推理任务上更快更稳。它出自 MiniMax-M1，是 2026 年 GRPO 家族里的主流变体之一。
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()  
            # 把重要性权重裁到上限内（裁权重，不是裁目标）。
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl) # #  ← 裁剪"重要性权重"，梯度只走 log π
        else:
            # loss = min( ratio·A,  clip(ratio, 1-ε, 1±ε)·A )
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)  # ← 裁剪"目标函数"

        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        # 反向传播，计算梯度
        loss.backward()

        # 梯度累积与优化器
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 间隔log_interval步，做一次统计
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # actor 权重转 `.half()` 存 `.pth`（推理/部署用 fp16）；再用 `lm_checkpoint` 存**完整状态**（含 critic、两个优化器、两个 scheduler、epoch/step），用于断点续训。注意 `_orig_mod` 是 torch.compile 包的一层壳，要先 `getattr` 剥掉再取 `state_dict`。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 最后 `del` 一大串张量：长训练里每步都 new 一堆大张量，显式释放避免"用到了下下下轮"，这属于长期训练的经验性写法。
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
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
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Reference模型和Policy模型是同一个
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 算优化器总共要更新几次
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    
    # 动态学习率 -> 余弦退火调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()