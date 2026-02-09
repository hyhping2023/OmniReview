import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import numpy as np

from models.mmoe import MMoE, confidence_predict_loss, recommendation_ranking_loss
from models.datasets import ReviewerTrainDataset, ReviewerTrainDataloader, ReviewerTestDataset, ReviewerTestDataloader
import pytrec_eval

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler.LambdaLR, device: str, task:int = 0) -> dict:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training data batches.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        device: Device to run the training on (e.g., "cuda:0").
        task: Task index (1 for confidence prediction, 2 for recommendation, 0 for both).

    Returns:
        A dictionary containing the average loss and other metrics.
    """
    model.train()
    total_loss = 0
    metrics_history = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue
        # 数据迁移到设备
        for key in ["emb_paper", "emb_conf", "label_conf", "emb_recommendation", "label_recommendation"]:
            batch[key] = batch[key].to(device)
        
        # 前向传播
        # task1: 正负样本置信度预测
        if task != 2:
            outputs_conf, _, conf_gates = model(
                emb_paper=batch["emb_paper"],
                emb_reviewer=batch["emb_conf"],
                task_idx=0
            )
            # 损失计算
            loss_conf = confidence_predict_loss(
                outputs=outputs_conf,
                batch={"label": batch["label_conf"]},
                gates=conf_gates,
                entropy_loss_weight=0.01
            )

        # task2: 推荐排序
        if task != 1:
            outputs_recommendation, _, rec_gates = model(
                emb_paper=batch["emb_paper"],
                emb_reviewer=batch["emb_recommendation"],
                task_idx=1
            )
        
            loss_recommendation = recommendation_ranking_loss(
                outputs=outputs_recommendation,
                batch={"label": batch["label_recommendation"]},
                margin=0.5,
                lambda_auc=0.5
            )

        if task == 1:
            loss = loss_conf
        elif task == 2:
            loss = loss_recommendation
        else:
            loss = loss_conf + loss_recommendation
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        if task == 1:
            torch.nn.utils.clip_grad_norm_(
                list(model.interaction_encoder.parameters()) + 
                list(model.multi_gating_network[0].parameters()) + 
                list(model.experts.parameters()) +
                list(model.multi_task_tower[0].parameters()),
                max_norm=1.0
            )
        elif task == 2:
            torch.nn.utils.clip_grad_norm_(
                list(model.multi_gating_network[1].parameters()) + 
                list(model.multi_task_tower[1].parameters()),
                max_norm=1.0
            )
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
                
        total_loss += loss.item()
        metrics_history.append({"loss": loss.item()})
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_metrics = {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0].keys()}
    return {"loss": total_loss / len(dataloader), **avg_metrics}

def validate(model: nn.Module, dataloader:DataLoader, device: str) -> tuple:
    """
    Validate the model on the validation dataset.

    Args:
        model: The neural network model to validate.
        dataloader: DataLoader providing validation data batches.
        device: Device to run the validation on (e.g., "cuda:0").

    Returns:
        A tuple containing average true confidence, average false confidence, and average NDCG.
    """
    model.eval()

    confidence_true_list = []
    confidence_false_list = []
    ndcg_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            for key in ["emb_paper", "emb_conf", "label_conf", "emb_recommendation", "label_recommendation"]:
                batch[key] = batch[key].to(device)
            
            # 前向传播
            # task1: 正负样本置信度预测
            outputs_conf, _, conf_gates = model(
                emb_paper=batch["emb_paper"],
                emb_reviewer=batch["emb_conf"],
                task_idx=0
            )

            # task2: 推荐排序
            outputs_recommendation, _, rec_gates = model(
                emb_paper=batch["emb_paper"],
                emb_reviewer=batch["emb_recommendation"],
                task_idx=1
            )

            # 计算真实审稿人得分和错误审稿人得分
            confidence_true = outputs_conf[batch["label_conf"] == 1].flatten().detach().cpu().numpy().tolist()
            confidence_false = outputs_conf[batch["label_conf"] == 0].flatten().detach().cpu().numpy().tolist()

            confidence_true_list.extend(confidence_true)
            confidence_false_list.extend(confidence_false)

            rec_true = outputs_recommendation[batch["label_recommendation"] == 1].detach().cpu().numpy().tolist()
            rec_false = outputs_recommendation[batch["label_recommendation"] == 0].detach().cpu().numpy().tolist()

            qrel = {'0': {}}
            run = {'0': {}}
            tempindex = 0
            for true_score in rec_true:
                tempindex += 1
                qrel['0'][str(tempindex)] = 1
                run['0'][str(tempindex)] = float(true_score)
            for false_score in rec_false:
                tempindex += 1
                run['0'][str(tempindex)] = float(false_score)

            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
            results = evaluator.evaluate(run)
            ndcg_list.append(results['0']['ndcg'])
    
    avg_true_confidence = np.mean(np.array(confidence_true_list))
    avg_false_confidence = np.mean(np.array(confidence_false_list))
    avg_ndcg = np.mean(np.array(ndcg_list))
    return avg_true_confidence, avg_false_confidence, avg_ndcg

def main():
    """
    Main training loop for the reviewer recommendation model.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda:0"
    save_path = './data/checkpoints/mmoe_new/3expert_new'
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Load datasets
    print("加载数据集...")
    train_dataset = ReviewerTrainDataset(df_path='path/to/train_dataset_qwen_sampled.parquet', sample_ratio=1)
    val_dataset = ReviewerTrainDataset(df_path='path/to/val_dataset_qwen_sampled.parquet')
    train_dataloader = ReviewerTrainDataloader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = ReviewerTrainDataloader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"训练集大小: {len(train_dataset)}  验证集大小: {len(val_dataset)}")

    # Initialize the model
    print("初始化模型...")
    model = MMoE(emb_dim=1024, num_experts=3, dropout=0.1, expert_size=512, 
                 num_tasks=2, router_noise=0.1, topk=-1)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        list(model.interaction_encoder.parameters()) + 
        list(model.multi_gating_network[0].parameters()) + 
        list(model.experts.parameters()) +
        list(model.multi_task_tower[0].parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    epoches = 100
    total_steps = len(train_dataloader) * epoches // 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_score = 0
    checkpoint_path = None

    print("Starting training...")
    for epoch in range(epoches//2):
        task = 1
        print(f"\n ============= Epoch {epoch + 1}/{epoches} =============")
        
        train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, device, task=task)
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        avg_true_confidence, avg_false_confidence, avg_ndcg = validate(model, val_dataloader, device)
        val_score = (avg_true_confidence + (1 - avg_false_confidence)) / 2
        print(f"Validation Avg True Confidence: {avg_true_confidence:.4f}")
        print(f"Validation Avg False Confidence: {avg_false_confidence:.4f}")
        print(f"Validation Avg NDCG: {avg_ndcg:.4f}")
        print(f"Validation Score: {val_score:.4f}")

        # Save the best model
        if best_score < val_score:
            best_score = val_score
            checkpoint_path = f"{save_path}/best_mmoe_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with val score: {best_score:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print("Starting second-stage training...")
    optimizer = torch.optim.AdamW(
        list(model.multi_gating_network[1].parameters()) + 
        list(model.multi_task_tower[1].parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    for epoch in range(epoches//2, epoches):
        task = 2
            
        print(f"\n ============= Epoch {epoch + 1}/{epoches} =============")
        
        # 训练
        train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, device, task=task)
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        avg_true_confidence, avg_false_confidence, avg_ndcg = validate(model, val_dataloader, device)
        val_score = (avg_true_confidence + (1 - avg_false_confidence)) / 2 + avg_ndcg
        print(f"Validation Avg True Confidence: {avg_true_confidence:.4f}")
        print(f"Validation Avg False Confidence: {avg_false_confidence:.4f}")
        print(f"Validation Avg NDCG: {avg_ndcg:.4f}")
        print(f"Validation Score: {val_score:.4f}")

        # 保存最佳模型
        if best_score < val_score:
            best_score = val_score
            checkpoint_path = f"{save_path}/best_mmoe_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with val score: {best_score:.4f}")

        # # 定期保存checkpoint
        # if (epoch + 1) % 5 == 0:
        #     checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch + 1}.pth"
        #     torch.save(model.state_dict(), checkpoint_path)

def test_evaluate():
    """测试评估函数"""
    device = "cuda:0"
    model = MMoE(emb_dim=1024, num_experts=3, dropout=0.1, expert_size=512, 
                 num_tasks=2, router_noise=0.1, topk=-1)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load('path/to/best/checkpoint', map_location=device, weights_only=True))
    
    test_dataset = ReviewerTestDataset()
    test_dataloader = ReviewerTestDataloader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        correct_score = {}
        wrong_score = {}
        similar_score = {}
        for inx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            inx = str(inx)
            temp_index = 0
            category = batch["category"][0]
            qrel = {inx:{}}
            true_scores = []
            similar_scores = []

            for key in ["emb_paper", "emb_reviewer", "emb_wrong", "emb_similar"]:
                batch[key] = batch[key].to(device)
            
            # Task1: 正确审稿人评分
            outputs_reviewer_conf, _, gates = model(
                batch["emb_paper"].expand(batch["emb_reviewer"].size(0), -1), 
                batch["emb_reviewer"],
                task_idx=0
            )
            print(gates)
            for o in outputs_reviewer_conf.flatten().detach().cpu().numpy().tolist():
                print(f"Final Cosine Similarity Score (True Reviewer): {o:.4f}  Category: {category}")

            outputs_wrong, _, gates = model(
                batch["emb_paper"].expand(batch["emb_wrong"].size(0), -1), 
                batch["emb_wrong"],
                task_idx=0
            )
            for o in outputs_wrong.flatten().detach().cpu().numpy().tolist():
                print(f"Final Cosine Similarity Score (Wrong Candidates): {o:.4f}  Category: {category}")

            # Task2: 推荐系统评分
            outputs_reviewer_recommend, _, gates = model(
                batch["emb_paper"].expand(batch["emb_reviewer"].size(0), -1), 
                batch["emb_reviewer"],
                task_idx=1
            )
            for o in outputs_reviewer_recommend.flatten().detach().cpu().numpy().tolist():
                print(f"Final Cosine Similarity Score (True Reviewerin Similar): {o:.4f}  Category: {category}")
                temp_index += 1
                qrel[inx][str(temp_index)] = 1
                true_scores.append((temp_index, o))

            outputs_similar, _, gates = model(
                batch["emb_paper"].expand(batch["emb_similar"].size(0), -1), 
                batch["emb_similar"],
                task_idx=1
            )
            for o in outputs_similar.flatten().detach().cpu().numpy().tolist():
                print(f"Final Cosine Similarity Score (Similar Reviewer): {o:.4f}  Category: {category}")
                temp_index += 1
                similar_scores.append((temp_index, o))
            
            all_candidates = true_scores + similar_scores
            print(all_candidates)
            print(f"All Candidates before sorting: {(len(all_candidates))}")
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            print(all_candidates)
            run = {inx: {str(cid): float(score) for cid, score in all_candidates}}
            print(run)
            print(qrel)
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg', 'map', 'recip_rank', 'success_5', 'Rprec'})
            results = evaluator.evaluate(run)
            print(f"Evaluation Metrics: {results[inx]}")
            for metric, value in results[inx].items():
                if category not in similar_score:
                    similar_score[category] = {}
                if metric not in similar_score[category]:
                    similar_score[category][metric] = []
                similar_score[category][metric].append(value)
            
        high_confidence_threshold = 0.9
        middle_confidence_threshold = 0.7
        low_confidence_threshold = 0.5
        for category, scores in correct_score.items():
            high_confidence = [s for s in scores if s > high_confidence_threshold]
            middle_confidence = [s for s in scores if middle_confidence_threshold < s <= high_confidence_threshold]
            low_confidence = [s for s in scores if s <= low_confidence_threshold]
            print(f"Category: {category}  High Confidence: {len(high_confidence)}  Middle Confidence: {len(middle_confidence)}  Low Confidence: {len(low_confidence)}")

        for category, scores in wrong_score.items():
            high_confidence = [s for s in scores if s < 1-high_confidence_threshold]
            middle_confidence = [s for s in scores if 1-high_confidence_threshold <= s < 1-middle_confidence_threshold]
            low_confidence = [s for s in scores if s >= 1 - low_confidence_threshold]
            print(f"(Wrong Reviewer) Category: {category}  High Confidence: {len(high_confidence)}  Middle Confidence: {len(middle_confidence)}  Low Confidence: {len(low_confidence)}")

        for category, metrics in similar_score.items():
            print(f"(Similar Reviewer) Category: {category}")
            for metric, values in metrics.items():
                avg_value = sum(values) / len(values)
                print(f"  Metric: {metric}  Average Value: {avg_value:.4f}")

if __name__ == "__main__":
    main()
    # test_evaluate()