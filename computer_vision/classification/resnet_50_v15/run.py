# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse
import os
from utils.benchmark import run_model
from utils.cv.imagenet import ImageNet
from utils.misc import print_goodbye_message_and_die, print_warning_message, download_ampere_imagenet


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 v1.5 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "bf16", "fp16", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["tf", "ort", "pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    parser.add_argument("--csv_path",
                        type=str,
                        default="",
                        help="path to csv file to save the result metrics"
                        )
    return parser.parse_args()


def run_tf(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, imagenet):
        shape = (224, 224)
        tf_runner.set_input_tensor("input_tensor:0", imagenet.get_input_array(shape))
        output = tf_runner.run(batch_size)
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["softmax_tensor:0"][i]),
                imagenet.extract_top5(output["softmax_tensor:0"][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)
    runner = TFFrozenModelRunner(model_path, ["softmax_tensor:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tflite import TFLiteRunner

    def run_single_pass(tflite_runner, imagenet):
        shape = (224, 224)
        tflite_runner.set_input_tensor(tflite_runner.input_details[0]['index'], imagenet.get_input_array(shape))
        tflite_runner.run(batch_size)
        output_tensor = tflite_runner.get_output_tensor(tflite_runner.output_details[0]['index'])
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output_tensor[i]),
                imagenet.extract_top5(output_tensor[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp(model_name, batch_size, num_runs, timeout, images_path, labels_path, precision = "fp32"):
    from utils.pytorch import PyTorchRunner
    import torch
    import torchvision

    def run_single_pass(pytorch_runner, imagenet):
        shape = (224, 224)
        if precision == "fp16":
            output = pytorch_runner.run(batch_size, torch.from_numpy(imagenet.get_input_array(shape)).half()).float()
        else:
            output = pytorch_runner.run(batch_size, torch.from_numpy(imagenet.get_input_array(shape))).float()

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing='PyTorch', is1001classes=False, order='NCHW')
    fp16 = (precision == "fp16")
    # import ipdb; ipdb.set_trace()
    runner = PyTorchRunner(torchvision.models.__dict__[model_name](pretrained=True),
                           disable_jit_freeze=False, fp16=fp16)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)

def run_pytorch_cuda(model_name, batch_size, num_runs, timeout, images_path, labels_path, disable_jit_freeze=False, **kwargs):
    from utils.pytorch import PyTorchRunner
    import torch
    import torchvision

    def run_single_pass(pytorch_runner, imagenet):
        shape = (224, 224)
        output = pytorch_runner.run(batch_size, torch.from_numpy(imagenet.get_input_array(shape)).cuda()).cpu()

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing='PyTorch', is1001classes=False, order='NCHW')
    runner = PyTorchRunner(torchvision.models.__dict__[model_name](pretrained=True).cuda(),
                           disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, imagenet):
        shape = (224, 224)
        ort_runner.set_input_tensor("input_tensor:0", imagenet.get_input_array(shape).astype("float16"))
        output = ort_runner.run(batch_size)

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[0][i]),
                imagenet.extract_top5(output[0][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True)
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_tf_bf16(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_pytorch_fp(model_name, batch_size, num_runs, timeout, images_path, labels_path)

def run_pytorch_fp16(model_name, batch_size, num_runs, timeout, images_path, labels_path, precision, **kwargs):
    return run_pytorch_fp(model_name, batch_size, num_runs, timeout, images_path, labels_path, precision)

def run_tflite_int8(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_ort_fp16(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def main():
    args = parse_args()
    download_ampere_imagenet()

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        elif args.precision == "bf16":
            run_tf_bf16(**vars(args))
        elif args.precision == "int8":
            run_tflite_int8(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    elif args.framework == "pytorch":
        import torch
        if torch.cuda.is_available():
            run_pytorch_cuda(model_name="resnet50", **vars(args))
        elif args.precision == "fp32":
            acc_res, perf_res = run_pytorch_fp32(model_name="resnet50", **vars(args))
        elif args.precision == "fp16":
            run_pytorch_fp16(model_name="resnet50", **vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    elif args.framework == "ort":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp16":
            run_ort_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)

    if args.csv_path:
        header = "aio_num_threads, batch_size, precision, top_1_acc, top_5_acc, " \
             "Latency_P50/s, Latency_P90/s, Latency_P99/s, Latency_AVG/s, QPS(req/s)\n"
        with open(args.csv_path, 'a') as f:
            if not f.tell():
                f.write(header)
            if os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] == ".*":
                precision = "fp16"
            else:
                precision = "fp32"
            line = f'{int(os.environ["AIO_NUM_THREADS"])},{args.batch_size},{precision},{acc_res["top_1_acc"]:.2f},' \
                f'{acc_res["top_5_acc"]:.2f},{perf_res["median_lat_ms"]:.3f},{perf_res["90th_percentile_lat_ms"]:.3f},' \
                f'{perf_res["99th_percentile_lat_ms"]:.3f},{perf_res["mean_lat_ms"]:.3f},{perf_res["observed_throughput_ips"]:.3f}\n'
            f.write(line)
            print_warning_message(f'Perf data have been saved in {args.csv_path}')

if __name__ == "__main__":
    main()
