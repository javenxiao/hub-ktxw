import os
import csv
import matplotlib.pyplot as plt

def save_training_logs(trainer, output_dir="training_logs"):
    """
    绘制训练曲线、保存曲线图片，并导出每 epoch 指标到 CSV

    Args:
        trainer: Hugging Face Trainer 对象
        output_dir: 保存文件目录
    """
    os.makedirs(output_dir, exist_ok=True)

    logs = trainer.state.log_history

    # 准备数据
    train_steps, train_loss = [], []
    val_steps, val_loss, val_acc, val_precision, val_recall, val_f1 = [], [], [], [], [], []

    for log in logs:
        # 训练 loss
        if "loss" in log and "epoch" in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])
        # 验证指标
        if "eval_loss" in log:
            val_steps.append(log["step"])
            val_loss.append(log.get("eval_loss", None))
            val_acc.append(log.get("eval_accuracy", None))
            val_precision.append(log.get("eval_precision", None))
            val_recall.append(log.get("eval_recall", None))
            val_f1.append(log.get("eval_f1", None))

    # 绘制 loss 曲线
    plt.figure(figsize=(10,4))
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.plot(val_steps, val_loss, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved to {loss_path}")

    # 绘制准确率曲线
    if any(val_acc):
        plt.figure(figsize=(10,4))
        plt.plot(val_steps, val_acc, label="Validation Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_path = os.path.join(output_dir, "accuracy_curve.png")
        plt.savefig(acc_path)
        plt.close()
        print(f"Accuracy curve saved to {acc_path}")

    # 保存每个 eval step 的指标到 CSV
    csv_file = os.path.join(output_dir, "eval_metrics.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
        writer.writerow(header)
        for i in range(len(val_steps)):
            writer.writerow([
                val_steps[i],
                val_loss[i],
                val_acc[i],
                val_precision[i],
                val_recall[i],
                val_f1[i]
            ])
    print(f"Evaluation metrics saved to {csv_file}")
