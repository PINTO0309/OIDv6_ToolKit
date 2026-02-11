#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop objects from images using YOLO-format annotations."
    )
    parser.add_argument(
        "--dataset-root",
        default="OID/Dataset",
        help="Dataset root directory (default: OID/Dataset)",
    )
    parser.add_argument(
        "--output-root",
        default="cropped_images",
        help="Output directory for cropped images (default: cropped_images)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Monkey"],
        help="Class folder names to process (default: Monkey)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to process (default: train validation test)",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap on images processed per split/class for quick tests",
    )
    return parser.parse_args()


def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    label = parts[0]
    try:
        x_center, y_center, width, height = map(float, parts[1:5])
    except ValueError:
        return None
    return label, x_center, y_center, width, height


def yolo_to_xyxy(x_center, y_center, width, height, image_w, image_h):
    x1 = int((x_center - width / 2.0) * image_w)
    y1 = int((y_center - height / 2.0) * image_h)
    x2 = int((x_center + width / 2.0) * image_w)
    y2 = int((y_center + height / 2.0) * image_h)

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(1, min(image_w, x2))
    y2 = max(1, min(image_h, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def find_annotation_file(class_dir, stem):
    same_dir = class_dir / f"{stem}.txt"
    if same_dir.exists():
        return same_dir

    label_dir = class_dir / "Label" / f"{stem}.txt"
    if label_dir.exists():
        return label_dir

    return None


def process_class_split(dataset_root, output_root, split, class_name, max_images_per_split=None):
    class_dir = dataset_root / split / class_name
    if not class_dir.exists():
        print(f"[WARN] Directory not found, skip: {class_dir}")
        return 0, 0, 0, 0

    images = [
        p for p in sorted(class_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if max_images_per_split is not None:
        images = images[:max_images_per_split]

    processed_images = 0
    missing_annotations = 0
    skipped_boxes = 0
    saved_crops = 0

    for image_path in images:
        annotation_path = find_annotation_file(class_dir, image_path.stem)
        if annotation_path is None:
            missing_annotations += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Cannot read image, skip: {image_path}")
            continue

        image_h, image_w = image.shape[:2]

        with annotation_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        box_index = 0
        for line in lines:
            parsed = parse_yolo_line(line)
            if parsed is None:
                skipped_boxes += 1
                continue

            _, x_center, y_center, width, height = parsed
            coords = yolo_to_xyxy(x_center, y_center, width, height, image_w, image_h)
            if coords is None:
                skipped_boxes += 1
                continue

            x1, y1, x2, y2 = coords
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                skipped_boxes += 1
                continue

            crop_name = f"{split}__{class_name}__{image_path.stem}__{box_index}.jpg"
            crop_path = output_root / crop_name
            ok = cv2.imwrite(str(crop_path), crop)
            if not ok:
                print(f"[WARN] Failed to save crop: {crop_path}")
                skipped_boxes += 1
                continue

            saved_crops += 1
            box_index += 1

        processed_images += 1

    return processed_images, missing_annotations, skipped_boxes, saved_crops


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    total_processed_images = 0
    total_missing_annotations = 0
    total_skipped_boxes = 0
    total_saved_crops = 0

    for split in args.splits:
        for class_name in args.classes:
            print(f"[INFO] Processing split={split}, class={class_name}")
            processed, missing, skipped, saved = process_class_split(
                dataset_root=dataset_root,
                output_root=output_root,
                split=split,
                class_name=class_name,
                max_images_per_split=args.max_images_per_split,
            )
            total_processed_images += processed
            total_missing_annotations += missing
            total_skipped_boxes += skipped
            total_saved_crops += saved
            print(
                f"[INFO] Done split={split}, class={class_name}: "
                f"images={processed}, missing_ann={missing}, skipped_boxes={skipped}, saved_crops={saved}"
            )

    print("[INFO] Summary")
    print(f"  processed_images: {total_processed_images}")
    print(f"  missing_annotations: {total_missing_annotations}")
    print(f"  skipped_boxes: {total_skipped_boxes}")
    print(f"  saved_crops: {total_saved_crops}")
    print(f"  output_root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
