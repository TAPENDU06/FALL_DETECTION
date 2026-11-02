
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_FILE = PROJECT_ROOT / "data/annotations/urfd_annotations.csv"
DATA_ROOT = PROJECT_ROOT / "data/raw/urfd"
OUTPUT_DIR = PROJECT_ROOT / "data/splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read annotations
df = pd.read_csv(ANNOTATION_FILE)

# Group annotations to obtain one label per video
videos = (
    df.groupby(["sequence_id", "camera_id"])["label"]
    .apply(lambda labels: "fall" if (labels == "fall").any() else "normal")
    .reset_index()
)

# Construct video paths

def get_video_path(row):
    sequence = row["sequence_id"]
    camera = row["camera_id"]
    label = row["label"]

    if label == "fall":
        folder = "Fall"
    else:
        folder = "No fall"

    filename = f"{sequence}-{camera}.mp4"

    return Path("D:/Fall_Dataset") / folder / "front_cam" / filename

videos["video_path"] = videos.apply(get_video_path, axis=1)

# Verify video files
missing = videos[~videos["video_path"].apply(Path.exists)]

if not missing.empty:
    print("Missing video files:")
    print(missing[["sequence_id", "camera_id", "video_path"]])
    raise FileNotFoundError("Fix missing video files before splitting.")

# Verify total videos
print("Total videos:", len(videos))
print("\nOverall class distribution:")
print(videos["label"].value_counts())

# First split: 70% training, 30% temporary
train, temporary = train_test_split(
    videos,
    test_size=0.30,
    random_state=42,
    stratify=videos["label"]
)

# Second split: 15% validation, 15% testing
validation, test = train_test_split(
    temporary,
    test_size=0.50,
    random_state=42,
    stratify=temporary["label"]
)

# Save split files
train.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Display results
for name, split in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:
    print(f"\n{name} SET")
    print("Total:", len(split))
    print(split["label"].value_counts())

print("\nDATASET SPLITTING COMPLETED")
print("Output directory:", OUTPUT_DIR)
