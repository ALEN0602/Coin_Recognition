import os
import random
import shutil

SOURCE_DIR = "dataset/raw"
OUTPUT_DIR = "dataset"

classes = [
    "1_piso",
    "5_piso",
    "10_piso",
    "20_piso",
    "25_centavo"
]

train_ratio = 0.70
val_ratio = 0.15

random.seed(42)

for split in ["train", "val", "test"]:

    split_folder = os.path.join(
        OUTPUT_DIR,
        split
    )

    if os.path.exists(split_folder):
        shutil.rmtree(split_folder)

        print(
            f"Removed old {split} folder"
        )

for cls in classes:

    source_folder = os.path.join(
        SOURCE_DIR,
        cls
    )

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(".jpg")
    ]

    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split, files in splits.items():

        output_folder = os.path.join(
            OUTPUT_DIR,
            split,
            cls
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        for file in files:

            src = os.path.join(
                source_folder,
                file
            )

            dst = os.path.join(
                output_folder,
                file
            )

            shutil.copy2(
                src,
                dst
            )

    print(
        f"{cls}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset split completed.")