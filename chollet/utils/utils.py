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


def tests_training_plots(tests_results):
    training_losses = [tests_results[test]["loss"] for test in tests_results]
    validation_losses = ([tests_results[test]["val_loss"]
                          for test in tests_results])
    training_acc = [tests_results[test]["acc"] for test in tests_results]
    validation_acc = [tests_results[test]["val_acc"] for test in tests_results]
    tests = [test for test in tests_results]

    plt.rcParams["figure.figsize"] = (12, 9)
    fig, ax = plt.subplots(2, 1)
    colors = ["b", "r", "g", "c", "m", "k"]
    colors = colors[:len(tests)]
    for i in range(0, len(tests)):        
        epochs = range(1, len(training_losses[i]) + 1)

        ax[0].plot(epochs, training_losses[i], colors[i] + "o",
                   label="Training loss - " + tests[i])
        ax[0].plot(epochs, validation_losses[i], colors[i],
                   label="Validation loss - " + tests[i])
        ax[0].set_title("Training and validation loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend(loc="upper right")

        ax[1].plot(epochs, training_acc[i], colors[i] + "o",
                   label="Traing accuracy - " + tests[i])
        ax[1].plot(epochs, validation_acc[i], colors[i],
                   label="Validation accuracy - " + tests[i])
        ax[1].set_title("Training and validation accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend(loc="upper right")
        plt.tight_layout
