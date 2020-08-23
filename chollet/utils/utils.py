import matplotlib.pyplot as plt


def training_plots(loss_values, val_loss_values, acc, acc_val):
    epochs = range(1, len(acc) + 1)

    plt.rcParams["figure.figsize"] = (12, 9)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(epochs, loss_values, "bo", label="Training loss")
    ax[0].plot(epochs, val_loss_values, "b", label="Validation loss")
    ax[0].set_title("Training and validation loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, acc, "bo", label="Traing accuracy")
    ax[1].plot(epochs, acc_val, "b", label="Validation accuracy")
    ax[1].set_title("Training and validation accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.tight_layout
