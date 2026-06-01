import argparse
import os
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

import fastflow

CLASSIFICATION_THRESHOLD = -0.15
IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PADDING_COLOR = (0, 0, 0) 


def build_model(config, checkpoint_path, device):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    image_files = []
    for extension in IMG_EXTENSIONS:
        image_files.extend(glob(os.path.join(input_path, "**", extension), recursive=True))
    return sorted(image_files)


def normalize_map(anomaly_map):
    anomaly_map = anomaly_map.astype(np.float32)
    map_min = anomaly_map.min()
    map_max = anomaly_map.max()
    if map_max - map_min < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
    return (anomaly_map * 255).clip(0, 255).astype(np.uint8)


def raw_map_to_uint8(anomaly_map):
    score_map = 1.0 + anomaly_map.astype(np.float32)
    return (score_map.clip(0.0, 1.0) * 255).astype(np.uint8)


def resize_and_pad(image, input_size):
    width, height = image.size
    scale = float(input_size) / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    padded = Image.new("RGB", (input_size, input_size), color=PADDING_COLOR)
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    padded.paste(resized, (left, top))
    content_box = (left, top, left + resized_width, top + resized_height)
    has_padding = resized_width != input_size or resized_height != input_size
    return padded, content_box, has_padding


def crop_to_content(anomaly_map, content_box):
    left, top, right, bottom = content_box
    return anomaly_map[top:bottom, left:right]


def make_heatmap(gray):
    x = gray.astype(np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha):
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    heatmap_np = heatmap.astype(np.float32)
    blended = image_np * (1.0 - alpha) + heatmap_np * alpha
    return blended.clip(0, 255).astype(np.uint8)


def make_segmentation(image, mask):
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_bool = mask > 0
    segmented = image_np * 0.35
    segmented[mask_bool] = image_np[mask_bool] * 0.45 + np.array([255, 32, 32]) * 0.55
    return segmented.clip(0, 255).astype(np.uint8)


def save_panel(image, heatmap, mask, overlay, segmentation, output_path):
    original = image.convert("RGB")
    panels = [
        original,
        Image.fromarray(heatmap),
        Image.fromarray(mask).convert("RGB"),
        Image.fromarray(overlay),
        Image.fromarray(segmentation),
    ]
    width, height = original.size
    canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel.resize((width, height)), (index * width, 0))
    canvas.save(output_path)


def infer_one(model, image_path, transform, input_size, output_dir, threshold, alpha, device):
    image = Image.open(image_path).convert("RGB")
    padded_image, content_box, has_padding = resize_and_pad(image, input_size)
    tensor = transform(padded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        ret = model(tensor)

    anomaly_map = ret["anomaly_map"][0, 0].detach().cpu().numpy()
    content_map = crop_to_content(anomaly_map, content_box)
    max_score = float(content_map.max())
    mean_score = float(content_map.mean())
    raw_score_map = raw_map_to_uint8(content_map)
    raw_score_map = np.asarray(
        Image.fromarray(raw_score_map).resize(image.size, Image.BILINEAR)
    )
    normalized_map = normalize_map(content_map)
    normalized_map = np.asarray(
        Image.fromarray(normalized_map).resize(image.size, Image.BILINEAR)
    )
    mask = ((normalized_map >= threshold) * 255).astype(np.uint8)
    heatmap = make_heatmap(normalized_map)
    overlay = overlay_heatmap(image, heatmap, alpha)
    segmentation = make_segmentation(image, mask)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    if has_padding:
        padded_image.save(os.path.join(output_dir, f"{stem}_padded_input.png"))
    Image.fromarray(raw_score_map).save(os.path.join(output_dir, f"{stem}_raw_score_map.png"))
    Image.fromarray(normalized_map).save(os.path.join(output_dir, f"{stem}_anomaly_map.png"))
    Image.fromarray(heatmap).save(os.path.join(output_dir, f"{stem}_heatmap.png"))
    Image.fromarray(mask).save(os.path.join(output_dir, f"{stem}_mask.png"))
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{stem}_overlay.png"))
    Image.fromarray(segmentation).save(os.path.join(output_dir, f"{stem}_segmentation.png"))
    save_panel(
        image,
        heatmap,
        mask,
        overlay,
        segmentation,
        os.path.join(output_dir, f"{stem}_panel.png"),
    )

    return max_score, mean_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastFlow inference and save visualizations")
    parser.add_argument("-cfg", "--config", type=str, required=True, help="path to config file")
    parser.add_argument("-ckpt", "--checkpoint", type=str, required=True, help="path to checkpoint")
    parser.add_argument("-i", "--input", type=str, required=True, help="image file or image folder")
    parser.add_argument("-o", "--output", type=str, default="fastflow_outputs", help="output folder")
    parser.add_argument("--threshold", type=int, default=128, help="binary mask threshold in [0, 255]")
    parser.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay alpha in [0, 1]")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="inference device")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"

    config = yaml.safe_load(open(args.config, "r"))
    input_size = config["input_size"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    model = build_model(config, args.checkpoint, args.device)
    image_files = collect_images(args.input)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {args.input}")

    for image_path in image_files:
        relative_dir = ""
        if os.path.isdir(args.input):
            relative_dir = os.path.relpath(os.path.dirname(image_path), args.input)
            if relative_dir == ".":
                relative_dir = ""
        output_dir = os.path.join(args.output, relative_dir)
        max_score, mean_score = infer_one(
            model,
            image_path,
            transform,
            input_size,
            output_dir,
            args.threshold,
            args.alpha,
            args.device,
        )
        good = max_score < CLASSIFICATION_THRESHOLD
        print(f"{image_path}: max={max_score:.1f}, mean={mean_score:.1f}, good={good}")


if __name__ == "__main__":
    main()
