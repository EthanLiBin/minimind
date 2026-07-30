import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    # start_step 从x步开始
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # 动态学习率，策略：余弦退火学习率调度
        # 这样设计的好处：训练到最后模型的"步长"很小，不会在最优解周围刹不住车，收敛更稳定。

        # 不要用固定学习率从头训到尾。打个比方：
        # 训练初期（预热）:  学习率大 → 步子大，快速找到好方向
        # 训练中期:         学习率中 → 稳定往最优解走
        # 训练末期:         学习率小 → 步子小，精细调整，不会在最优解附近"跳来跳去"

        # 把当前是第几步和总共需要多少步塞进余弦公式，算出这节课应该用多大学习率，每个 step 都重新算一次。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # autocast_ctx 在 GPU 上用混合精度加速训练，在 CPU 上什么都不做。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss -> 语言模型 loss（预测下一个词差了多远），必须有的
            # res.aux_loss -> MoE 负载均衡 loss，非 MOE 模型时是 0
            loss = res.loss + res.aux_loss
            # 假设 batch_size=4, accumulation_steps=4，又因为一张卡显存放不下 16 条数据，只放得下 4 条。
            # 梯度累积的做法是分 4 步算，每一步算 4 条，合起来效果等价于 batch_size=16：
            # 因为显存放不下大 batch，就分批算梯度，累加够了再统一更新。除以 accumulation_steps 是为了让累加后的梯度和一口气跑大 batch 一模一样，否则学习率会放大 N 倍。
            loss = loss / args.accumulation_steps
            # 合起来一起 backward，模型既学预测又学均匀分配专家

        # fp16 梯度太小会下溢变 0。scaler.scale 先给 loss 乘一个大数（比如 65536），再反向传播：
        scaler.scale(loss).backward()

        """每个微 batch:
            loss/4 → scaler.scale → backward() → 梯度累加(已放大)

            每 4 步触发:
            ┌─ scaler.unscale_   →  梯度缩回真实值 + NaN检测
            ├─ clip_grad_norm_   →  梯度裁限，别炸
            ├─ scaler.step       →  优化器更新参数（如有溢出就跳过）
            ├─ scaler.update     →  动态调整下次缩放因子
            └─ zero_grad         →  清空梯度，准备下一轮

            一句话总结：放大→backward→缩回→裁剪→更新→调因子→清零，七个步骤环环相扣，都是在处理 fp16 的精度边界问题。
        """
        # accumulation_steps -> 每攒够 N 步的小梯度，才对参数真正更新一次。
        if step % args.accumulation_steps == 0:
            # scaler.unscale_ -> 把刚才放大的梯度除回去，恢复到正确的大小
            scaler.unscale_(optimizer)
            # 梯度裁剪。限制梯度的大小，不要让它爆炸，总长度限制在 grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # optimizer.step()  # 每个参数 = 参数 - lr × 梯度
            # 如果有溢出，这步直接跳过不更新。
            scaler.step(optimizer)
            # 更新 scaler 内部的缩放因子
            # 缩放因子在动态调整：没有溢出就慢慢放大，出错了就立刻缩小。
            scaler.update()

            # 清空累加的梯度，下一轮从零开始：
            optimizer.zero_grad(set_to_none=True)

        # 每隔 N 步反馈一次训练的"生命体征"——跑了多久、loss 多大、学习率多少、还有多久结束。让你不用盯着终端也能知道模型在不在正常学到东西。
        # step == iters -> (step == 最后一步)
        if step % args.log_interval == 0 or step == iters:
            # 计算耗时
            spend_time = time.time() - start_time
            # 还原真实的 loss，这样日志记录的是一整个 batch(含累积)的实际 loss，不是那 1/4 mini 的
            current_loss = loss.item() * args.accumulation_steps
            # 拆开 aux_loss
            # 总 loss = 预测误差(logits_loss) + 专家均衡惩罚(aux_loss)
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 预估剩余时间（ETA）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # 打印日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 上报到监控平台
            # 如果有 wandb（实验追踪工具），把这些指标推到云端，可以用来看训练曲线——loss 有没有下降、lr 对不对、aux_loss 有没有爆炸。
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
        
        """每 save_interval 步:
            ┌─ 只在 rank=0 的卡上执行
            ├─ eval()  ← 临时关 dropout
            ├─ 穿透 DDP + compile 包装
            ├─ 存权重 (.half().cpu())  → 推理用
            ├─ 存完整 checkpoint       → 断点续训用
            ├─ train() ← 恢复训练模式
            └─ del 释放内存"""
        # (每save_interval步/step==最后一步) and 只在主进程的时候触发
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切到推理模式
            model.eval()
            # 构建文件名
            # MOE模型:   ./out/pretrain_768_moe.pth
            # 普通模型:  ./out/pretrain_768.pth
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 穿透 DDP 和 torch.compile 的包装
            # 第一层剥 DDP:          model.module → 拿到被 DDP 包裹的模型
            # 第二层剥 compile:      _orig_mod    → 拿到编译前的原始模型
            # 最终拿到裸的 MiniMindForCausalLM，才能挂取 state_dict()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 保存权重（fp16 + CPU）
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 完整 checkpoint（用于断点续训）
            # 这个存的是完整训练状态
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 回到训练模式
            model.train()
            # 释放临时内存
            # state_dict 会把整套模型的 fp16 副本挂内存，保存完立刻删掉，别占着不用的内存。
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        # scaler.unscale_ -> 把刚才放大的梯度除回去，恢复到正确的大小
        scaler.unscale_(optimizer)
        # 梯度裁剪。限制梯度的大小，不要让它爆炸，总长度限制在 grad_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # optimizer.step()  # 每个参数 = 参数 - lr × 梯度
        # 如果有溢出，这步直接跳过不更新。
        scaler.step(optimizer)
        # 更新 scaler 内部的缩放因子
        # 缩放因子在动态调整：没有溢出就慢慢放大，出错了就立刻缩小。
        scaler.update()
        # 清空累加的梯度，下一轮从零开始：
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 统一设置随机种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 训练过程会自动在 ./checkpoints/ 目录保存完整检查点（模型、优化器、训练进度等）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # autocast 在 GPU 上用混合精度加速训练，在 CPU 上什么都不做。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # wandb 实时可视化监控
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器：多 GPU 训练时，每张卡不能看到完全一样的数据，否则就浪费了并行计算。每张卡看不同的数据，合起来才是一个完整 batch。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 混合精度训练缩放器：只对 float16 启用。fp16 的数值范围比 fp32 小很多（最小能表示 6e-5 左右），梯度太小的话会下溢变成 0，参数就学不动了。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 分布式：DistributedSampler 负责分数据，DistributedDataParallel（DDP）负责同步梯度
    # 每张卡虽然拿到的数据不同，但 gradient 通过 AllReduce 通信求平均后，每张卡得到的是完全一样的平均梯度。然后各自 optimizer.step() 更新的幅度一样，所以所有卡上的模型参数始终保持同步。
    # DataParallel 是分数据、AllReduce 同步梯度；模型权重在每个进程上都有全量拷贝，更新后完全一致。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置 epoch — 让每个 epoch 的数据分布不同
        # 如果 train_sampler 是 None（单卡），后面不执行；如果是分布式，set_epoch(epoch) 让 sampler 换一个随机种子，这样每个 epoch 每张卡分到的数据子集不一样，避免模型每次看到完全相同的顺序。
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个 epoch 的全局随机种子都不一样（42, 43, 44...），保证可复现又每个 epoch 有变化。randperm 生成随机排列的索引，单卡时用这个来打乱数据。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训的跳过逻辑，如果从 checkpoint 恢复后还是同一个 epoch（比如训练到 500 步 crash 了），跳过已经训练过的 500 步，从第 501 步开始
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 跳过采样器
        # 把"跳过"这件事内置到采样器里。首轮直接取 [skip*batch_size:] 之后的数据，跳过 skip 个 batch
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 实际训练
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()