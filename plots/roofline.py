import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == "__main__":
    flops_per_byte = [
        113.554050073427,
        123.047220124122,
        126.685632085377,
        128.26429842976,
        129.148030101639,
    ]
    gflops_per_second = [
        39644.6889151966,
        51160.3206192681,
        56172.8004292086,
        59011.4367299136,
        59325.3374649655,
    ]
    fig, ax = plt.subplots()
    ax.axhline(
        y=155900, xmin=0, xmax=150, color="r", linestyle="dashed", label="Max GFLOPS"
    )
    ax.plot(flops_per_byte, gflops_per_second, marker="o", label="VGG16 Performance")
    ax.set_xlabel("Operational Intensity (FLOPs/Byte)")
    ax.set_ylabel("GFLOPs/s")
    ax.set_title("Roofline Model")
    ax.legend()
    fig.savefig("roofline.png")
