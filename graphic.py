import matplotlib.pyplot as plt
import re
import os

def cargar_train_test_logs(path):
    epochs = []
    train_losses = []
    train_accs = []
    test_accs = []

    regex = re.compile(
        r"Epoch\s+(\d+),\s+Train Loss:\s+([0-9.eE+-]+),\s+Train Acc:\s+([0-9.]+)%,\s+Train Time:\s+[0-9.eE+-]+s,\s+Test Acc:\s+([0-9.]+)%",
        re.IGNORECASE
    )

    with open(path, 'r') as f:
        for line in f:
            match = regex.match(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_acc = float(match.group(3)) / 100
                test_acc = float(match.group(4)) / 100

                epochs.append(epoch)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

    return epochs, train_losses, train_accs, test_accs

# RUTA a tu archivo
log_path = "./output/FASHION/log.txt"

if not os.path.exists(log_path):
    print(f"Archivo no encontrado: {log_path}")
    exit()

# Cargar datos del log
epochs, train_losses, train_accs, test_accs = cargar_train_test_logs(log_path)

# ========================
# GRÁFICO 1: Loss + Train Acc
# ========================
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Entrenamiento y Evaluación", fontsize=14)

axs[0].plot(epochs, train_losses, label="Train Loss", color="red", linewidth=2)
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss", color="red")
axs[0].tick_params(axis='y', labelcolor="red")
axs[0].set_title("Training Loss / Accuracy")
axs[0].grid(True)

# Agregar Train Accuracy con segundo eje
ax2 = axs[0].twinx()
ax2.plot(epochs, train_accs, label="Train Accuracy", color="blue", linestyle="--", linewidth=2)
ax2.set_ylabel("Accuracy", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")
ax2.set_ylim(0, 1.05)

# Leyenda combinada
lines, labels = axs[0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[0].legend(lines + lines2, labels + labels2, loc="upper right")

# ========================
# GRÁFICO 2: Train vs Test Accuracy
# ========================
axs[1].plot(epochs, train_accs, label="Train Accuracy", color="blue", linestyle="--", linewidth=2)
axs[1].plot(epochs, test_accs, label="Test Accuracy", color="green", linewidth=2)
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Train/Test Accuracy")
axs[1].grid(True)

# Línea vertical en mejor Test Accuracy
best_epoch = epochs[test_accs.index(max(test_accs))]
best_acc = max(test_accs)

axs[1].axvline(x=best_epoch, linestyle='--', color='gray',
               label=f"Best Test Acc ({best_acc*100:.2f}%) @ Epoch {best_epoch}")
axs[1].annotate(f"{best_acc*100:.2f}%", xy=(best_epoch, best_acc),
                xytext=(best_epoch+1, best_acc - 0.03),
                arrowprops=dict(arrowstyle="->", color='gray'), fontsize=10)

# Ajustar zoom dinámico al rango de acc
combined_accs = train_accs + test_accs
acc_min2 = min(combined_accs)
acc_max2 = max(combined_accs)
acc_margin2 = 0.02
axs[1].set_ylim(max(0, acc_min2 - acc_margin2), min(1, acc_max2 + acc_margin2))

axs[1].legend(loc="lower right")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()