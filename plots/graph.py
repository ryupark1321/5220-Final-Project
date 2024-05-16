import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == "__main__":
    training_gpu_numbers = ["1", "2", "3", "4"]
    training_gpu_scaling = [912.938, 560.7268500328064, 425.0933737754822, 354.33]
    training_batch_sizes = ["128", "256", "512"]
    training_batch_scaling = [354.33, 329.7, 297.27]
    # training_weak = np.log2(np.array([6500, 12610, 25813, 51893]))
    training_weak = [6500, 12610, 25813, 51893]
    weak_scaling = [306.29, 504.47, 953.66, 1837.99]
    fig, ax = plt.subplots()
    y_offset = 0.1
    width = 0.6
    bottom = np.zeros(4)
    plt.bar(training_gpu_numbers, training_gpu_scaling, width)
    for i, total in enumerate(training_gpu_scaling):
        ax.text(i, total + y_offset, round(total, 3), ha="center", weight="bold")
    plt.xlabel("Number GPUs ")
    plt.ylabel("Training Time (s)")
    plt.legend()
    ax.set_title("Strong Scaling, Global Batch Size = 128 ")

    plt.savefig("gpu_scaling.png")

    plt.plot(training_weak, weak_scaling, linestyle="--", marker="o")
    # fig, ax = plt.subplots()
    # y_offset = 0.1
    # width = 0.6
    # bottom = np.zeros(4)
    # plt.bar(training_weak, weak_scaling, width)
    # for i, total in enumerate(weak_scaling):
    #     ax.text(i, total + y_offset, round(total, 3), ha="center", weight="bold")
    plt.xlabel("Number Training Examples")
    plt.ylabel("Training Time (s)")
    plt.legend()
    plt.title("Weak Scaling, Global Batch Size = 512")

    # plt.savefig("weak_scaling.png")

    # fig, ax = plt.subplots()
    # y_offset = 0.1
    # width = 0.6
    # bottom = np.zeros(3)
    # plt.bar(training_batch_sizes, training_batch_scaling, width)
    # for i, total in enumerate(training_batch_scaling):
    #     ax.text(i, total + y_offset, round(total, 3), ha="center", weight="bold")
    # plt.xlabel("Global Batch Size")
    # plt.ylabel("Training Time (s)")
    # plt.legend()
    # ax.set_title("Batch Size Scaling")

    # plt.savefig("batch_scaling.png")
