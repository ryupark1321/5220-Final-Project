def calculate_conv_flops_with_relu(H, W, C, K, out_channels):
    """
    Args:
        H (int): Height of the input feature map.
        W (int): Width of the input feature map.
        C (int): Number of input channels.
        K (int): Size of the convolutional kernel (square).
        out_channels (int): Number of output channels.
    """
    conv_flops = (H - K + 1) * (W - K + 1) * K * K * C * out_channels
    relu_flops = (H - K + 1) * (W - K + 1) * out_channels
    return conv_flops + relu_flops


def calculate_fc_flops_with_activation(input_size, output_size):
    """
    Args:
        input_size (int): Number of neurons in the input layer.
        output_size (int): Number of neurons in the output layer.
    """
    mul_flops = 2 * input_size * output_size
    total_flops = mul_flops + output_size + output_size
    return total_flops


def calculate_pooling_flops(H, W, pooling_size, stride):
    """
    Args:
        H (int): Height of the input feature map.
        W (int): Width of the input feature map.
        pooling_size (int): Size of the square pooling window.
        stride (int): Stride of the pooling operation.
    """
    output_H = (H - pooling_size) // stride + 1
    output_W = (W - pooling_size) // stride + 1
    return output_H * output_W * pooling_size * pooling_size


flops = 0

flops += calculate_conv_flops_with_relu(H=224, W=224, C=3, K=3, out_channels=64)
flops += calculate_conv_flops_with_relu(H=224, W=224, C=64, K=3, out_channels=64)
flops += calculate_pooling_flops(H=224, W=224, pooling_size=2, stride=2)

flops += calculate_conv_flops_with_relu(H=112, W=112, C=64, K=3, out_channels=128)
flops += calculate_conv_flops_with_relu(H=112, W=112, C=128, K=3, out_channels=128)
flops += calculate_pooling_flops(H=112, W=112, pooling_size=2, stride=2)

flops += calculate_conv_flops_with_relu(H=56, W=56, C=128, K=3, out_channels=256)
flops += calculate_conv_flops_with_relu(H=56, W=56, C=256, K=3, out_channels=256)
flops += calculate_conv_flops_with_relu(H=56, W=56, C=256, K=3, out_channels=256)
flops += calculate_pooling_flops(H=56, W=56, pooling_size=2, stride=2)

flops += calculate_conv_flops_with_relu(H=28, W=28, C=256, K=3, out_channels=512)
flops += calculate_conv_flops_with_relu(H=28, W=28, C=512, K=3, out_channels=512)
flops += calculate_conv_flops_with_relu(H=28, W=28, C=512, K=3, out_channels=512)
flops += calculate_pooling_flops(H=28, W=28, pooling_size=2, stride=2)

flops += calculate_conv_flops_with_relu(H=14, W=14, C=512, K=3, out_channels=512)
flops += calculate_conv_flops_with_relu(H=14, W=14, C=512, K=3, out_channels=512)
flops += calculate_conv_flops_with_relu(H=14, W=14, C=512, K=3, out_channels=512)
flops += calculate_pooling_flops(H=14, W=14, pooling_size=2, stride=2)

flops += calculate_fc_flops_with_activation(input_size=7 * 7 * 512, output_size=4096)
flops += calculate_fc_flops_with_activation(input_size=4096, output_size=4096)
flops += calculate_fc_flops_with_activation(input_size=4096, output_size=1000)

print(f"Total FLOPs: {flops}")
