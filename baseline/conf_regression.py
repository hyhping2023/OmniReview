import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression   # Platt scaling
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

sns.set_style("whitegrid")

# ---------- 1. 训练 ----------
def train_calibration(scores, labels, save_prefix="calib"):
    """
    训练两种校准器：保序回归 & Platt scaling
    返回 dict，含模型和评测指标
    """
    scores = scores.reshape(-1, 1) if scores.ndim == 1 else scores
    labels = labels.astype(int)

    # --- 1.1 Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(scores.squeeze(), labels)
    prob_iso = iso.transform(scores.squeeze())

    # --- 1.2 Platt (LR) ---
    lr = LogisticRegression(C=1e6, max_iter=1000)  # C很大≈无正则
    lr.fit(scores, labels)
    prob_platt = lr.predict_proba(scores)[:, 1]

    # --- 1.3 评测 ---
    results = {
        "iso": {
            "model": iso,
            "brier": brier_score_loss(labels, prob_iso),
            "logloss": log_loss(labels, prob_iso),
            "prob": prob_iso,
            "scores": scores.squeeze(),
        },
        "platt": {
            "model": lr,
            "brier": brier_score_loss(labels, prob_platt),
            "logloss": log_loss(labels, prob_platt),
            "prob": prob_platt,
        },
        "uncalib": {
            # 把原始 score 线性拉伸到 0~1 作为 baseline
            "prob": (scores.squeeze() - scores.min()) / (scores.max() - scores.min() + 1e-8),
            "brier": brier_score_loss(labels, (scores.squeeze() - scores.min()) / (scores.max() - scores.min() + 1e-8)),
            "logloss": log_loss(labels, (scores.squeeze() - scores.min()) / (scores.max() - scores.min() + 1e-8)),
        },
    }

    # 保存模型
    import joblib
    joblib.dump(iso, f"{save_prefix}_iso.gz")
    joblib.dump(lr, f"{save_prefix}_platt.gz")
    print("模型已保存到 disk：", f"{save_prefix}_iso.gz / _platt.gz")
    return results


# ---------- 2. 可靠性曲线 + ECE ----------
def reliability_diagram(labels, prob, n_bins=10, title_suffix=""):
    """
    计算可靠性曲线和 ECE
    """
    bin_true, bin_pred = calibration_curve(labels, prob, n_bins=n_bins)
    ece = np.mean(np.abs(bin_true - bin_pred))  # 简化版 ECE
    plt.plot(bin_pred, bin_true, marker='o', label=f"{title_suffix} (ECE={ece:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram")
    return ece


# ---------- 3. 可视化 ----------
def visualize(results, labels, save_path="calibration.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3.1 可靠性曲线
    plt.sca(axes[0])
    for k in ["uncalib", "platt", "iso"]:
        reliability_diagram(labels, results[k]["test_prob"], title_suffix=k)
    plt.legend()

    # 3.2 校准前后概率分布
    plt.sca(axes[1])
    for k in ["uncalib", "platt", "iso"]:
        sns.histplot(results[k]["test_prob"], bins=30, kde=True, label=k, alpha=0.6)
    plt.xlabel("Predicted probability")
    plt.title("Probability distribution")

    # 3.3 校准前后 vs 原始 score 散点
    plt.sca(axes[2])
    for k in ["uncalib", "iso", 'platt']:          # 想画几种就写几种
        plt.scatter(results[k]["test_scores"],
                    results[k]["test_prob"],
                    alpha=0.1, s=1, label=k)
    plt.xlabel("Original score")
    plt.ylabel("Calibrated probability")
    plt.title("Score → Probability mapping (test set)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ---------- 4. 主流程 ----------
def main():
    # 4.1 读数据
    rng = np.random.RandomState(0)
    correct_scores = np.load('embedding/merged/correct_scores.npy')[:, 0]   # 取第一列
    incorrect_scores = np.load('embedding/merged/wrong_scores.npy')[:, 0]
    print("Loaded scores:", correct_scores.shape, incorrect_scores.shape)

    # ---- 训练集：各采样 100 k ----
    # train_n = 100_000
    # idx_c = rng.choice(len(correct_scores), size=train_n, replace=False)
    # idx_w = rng.choice(len(incorrect_scores), size=train_n, replace=False)
    # 分析全部数据
    print("correct_scores:", correct_scores.mean(), correct_scores.std(), correct_scores.min(), correct_scores.max())
    print("incorrect_scores:", incorrect_scores.mean(), incorrect_scores.std(), incorrect_scores.min(), incorrect_scores.max())
    train_scores = np.concatenate([correct_scores, incorrect_scores])
    train_labels = np.concatenate([np.ones(len(correct_scores)), np.zeros(len(incorrect_scores))])
    print(train_scores.shape, train_labels.shape)
    # 绘制原始数据散点图, correct和incorrect的颜色区分开
    plt.figure(figsize=(6, 6))
    plt.scatter(range(len(incorrect_scores)), incorrect_scores, color='red', alpha=0.5, label='Incorrect Scores', s=2)
    plt.scatter(range(len(correct_scores)), correct_scores, color='blue', alpha=0.5, label='Correct Scores', s=2)
    plt.xlabel('Index')
    plt.ylabel('Scores')
    plt.title('Scatter Plot of Correct and Incorrect Scores')
    plt.legend()
    plt.savefig('score_scatter_plot.png', dpi=150)

    # 4.2 训练校准器（只用训练集）
    res = train_calibration(train_scores, train_labels)

    # ---- 测试集：全部数据 ----
    test_scores = np.concatenate([correct_scores, incorrect_scores])
    test_labels = np.concatenate([np.ones(len(correct_scores)),
                                  np.zeros(len(incorrect_scores))])

    # 4.3 在测试集上重新计算指标
    for k in res:
        if k == "uncalib":
            prob = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)
        else:
            prob = res[k]["model"].predict_proba(test_scores.reshape(-1, 1))[:, 1] \
                   if k == "platt" else \
                   res[k]["model"].transform(test_scores)
        res[k]["test_brier"] = brier_score_loss(test_labels, prob)
        res[k]["test_logloss"] = log_loss(test_labels, prob)
        res[k]["test_prob"] = prob          # 存下来画图用
        res[k]["test_scores"] = test_scores  # 保存原始 test score，避免 KeyError

    # 打印测试集指标
    print("\n======== 测试集指标 ========")
    for k in ["uncalib", "platt", "iso"]:
        print(f"{k:8s}  Brier={res[k]['test_brier']:.4f}  "
              f"LogLoss={res[k]['test_logloss']:.4f}")

    # 4.4 可视化（用测试集概率）
    visualize(res, test_labels, save_path="calibration_test.png")


if __name__ == "__main__":
    main()
