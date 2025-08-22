import os

# Path to images folder (adjust if needed)
img_dir = os.path.join(os.getcwd(), "images")

if not os.path.exists(img_dir):
    print("❌ images/ folder not found. Run collect_data.py first.")
else:
    # Extract names from filenames (everything before "_")
    people = sorted(set([f.split("_")[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]))
    
    if people:
        print("✅ People in dataset:")
        for p in people:
            print(" -", p)
    else:
        print("⚠️ No face images found in images/ folder.")
