import os
import shutil

# Paths
img_dir = os.path.join(os.getcwd(), "images")
data_dir = os.path.join(os.getcwd(), "data")
model_file = os.path.join(os.getcwd(), "final_model.h5")
best_model_file = os.path.join(os.getcwd(), "best_model.h5")
encoder_file = os.path.join(os.getcwd(), "label_encoder.pkl")

def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"🗑️ Deleted folder: {path}")
    else:
        print(f"⚠️ Folder not found: {path}")

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Deleted file: {path}")
    else:
        print(f"⚠️ File not found: {path}")

if __name__ == "__main__":
    print("⚠️ WARNING: This will delete ALL face data and trained models.")
    confirm = input("Type 'YES' to continue: ")

    if confirm == "YES":
        delete_folder(img_dir)
        delete_folder(data_dir)
        delete_file(model_file)
        delete_file(best_model_file)
        delete_file(encoder_file)

        print("\n✅ Project reset complete. All face data and trained models removed.")
        print("👉 You can now re-run collect_data.py to build a fresh dataset.")
    else:
        print("❌ Operation cancelled.")
