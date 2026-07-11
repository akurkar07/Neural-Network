import time
from pathlib import Path

import numpy as np

from dependencies import cost


def iter_batches(inputs, outputs, batch_size):
    """Yields matching input/output slices of up to batch_size examples."""
    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        yield start, inputs[start:end], outputs[start:end]


def save_metric_plot(values, plot_path, y_label):
    """Saves an epoch-based training metric plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(values) + 1)
    plt.figure()
    plt.plot(epochs, values)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(plot_path)
    plt.close()


def evaluate_network(network, test_inputs, test_outputs, test_y, verbose=False):
    """Tests the network and returns accuracy, wrong count, and average cost."""
    total_cost = 0
    wrong = 0

    if verbose:
        for count, normalised_input in enumerate(test_inputs):
            desired = test_outputs[count]
            answer = network.forwardPass(normalised_input)
            prediction = int(np.argmax(answer))
            if prediction != test_y[count]:
                wrong += 1
            example_cost = cost(desired, answer)
            total_cost += example_cost
            print(
                f"Given an image, NN returned {prediction}. "
                f"That should be {test_y[count]}. Cost of that example was {example_cost}"
            )
    else:
        evaluation_batch_size = 512
        for start, batch_inputs, batch_outputs in iter_batches(test_inputs, test_outputs, evaluation_batch_size):
            answers = network.forwardBatch(batch_inputs)
            predictions = np.argmax(answers, axis=1)
            batch_labels = test_y[start:start + len(predictions)]
            wrong += int(np.count_nonzero(predictions != batch_labels))
            total_cost += cost(batch_outputs, answers)

    no_of_examples = len(test_inputs)
    correct = no_of_examples - wrong
    accuracy = correct * 100 / no_of_examples
    average_cost = total_cost / no_of_examples
    return accuracy, wrong, average_cost


def train_network(
    network,
    train_inputs,
    train_outputs,
    epochs,
    learning_rate,
    batch_size,
    progress=False,
    test_inputs=None,
    test_outputs=None,
    test_y=None,
):
    """Trains a network and returns timing and cost statistics."""
    lowest_cost = float("inf")
    average_costs = []
    accuracies = []
    test_average_costs = []
    wrong_counts = []
    epoch_seconds = []
    updates = 0
    started_at = time.perf_counter()

    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        total_cost = 0

        for start, batch_inputs, batch_outputs in iter_batches(train_inputs, train_outputs, batch_size):
            if progress and start % 10000 == 0:
                print(f"Training example {start}")

            outputs = network.forwardBatch(batch_inputs)
            total_cost += cost(outputs, batch_outputs)
            network.backwardBatch(batch_outputs, learning_rate)
            updates += 1

        average_cost = total_cost / len(train_inputs)
        average_costs.append(float(average_cost))
        lowest_cost = min(total_cost, lowest_cost)
        epoch_seconds.append(time.perf_counter() - epoch_started_at)

        if test_inputs is not None:
            accuracy, wrong, test_average_cost = evaluate_network(network, test_inputs, test_outputs, test_y)
            accuracies.append(accuracy)
            wrong_counts.append(wrong)
            test_average_costs.append(float(test_average_cost))
            if progress:
                print(f"Epoch {epoch}, Test accuracy: {accuracy:.2f}%")

        if progress and epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Total Cost: {total_cost}, "
                f"Average cost/example: {average_cost}, LR: {learning_rate}, Batch size: {batch_size}"
            )

    total_seconds = time.perf_counter() - started_at
    return {
        "lowest_cost": lowest_cost,
        "average_costs": average_costs,
        "accuracies": accuracies,
        "test_average_costs": test_average_costs,
        "wrong_counts": wrong_counts,
        "epoch_seconds": epoch_seconds,
        "total_seconds": total_seconds,
        "updates": updates,
    }
