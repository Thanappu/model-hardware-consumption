#vlms_result = vlms.predict(img)
#result = zero_shot.predict(vlms_result['response'])
import os
import cv2
import pandas as pd
import src.model.vlms.vit_gpt2 as wedo_vit
import src.model.vlms.blip_large as wedo_blip_large
import src.model.vlms.blip_base as wedo_blip_base
import src.model.zeroshot_classif.deberta_v3_large as wedo_zeroshot
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

vlms = wedo_vit.VlmsVit()
zero_shot = wedo_zeroshot.ZeroshotDebertaLarge()

dataset_path = '/Volumes/pond/model_evaluate/image_dataset/office_classification/test'
output_csv = f"/Users/thanapoompumee/Documents/project/file/file_evaluate_models/{vlms.name}_evaluation_results.csv"
confusion_matrix_csv = output_csv.replace(".csv", "_confusion_matrix.csv")

data = []
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, filename)
                data.append([img_path, class_name, ""])

df = pd.DataFrame(data, columns=["Image_Path", "Label", f"{vlms.name}"])

for index, row in df.iterrows():
    img_path = row["Image_Path"]
    img = cv2.imread(img_path)
    if img is not None:
        vlms_result = vlms.predict(img)
        df.at[index, f"{vlms.name}"] = vlms_result.get("response", "N/A")

zero_shot_labels = zero_shot.package_info["Category"]
print(f"Zero-Shot Categories: {zero_shot_labels}")
df[f"ZeroShot_Output_{vlms.name}"] = ""
df[f"ZeroShot_Score_{vlms.name}"] = ""
df[f"Correct_{vlms.name}"] = 0

for index, row in df.iterrows():
    text_input = row[f"{vlms.name}"]

    if not text_input or text_input == "N/A":
        print(f"Warning: No text from VLMs ({vlms.name}) at index {index}")
        df.at[index, f"ZeroShot_Output_{vlms.name}"] = "N/A"
        df.at[index, f"ZeroShot_Score_{vlms.name}"] = 0.0
        continue

    print(f"Running Zero-Shot for {vlms.name} | Text: {text_input} | Labels: {zero_shot_labels}")
    zero_shot_result = zero_shot.predict(text_input)
    predicted_label = zero_shot_result.get("Output", "N/A")
    df.at[index, f"ZeroShot_Output_{vlms.name}"] = predicted_label
    df.at[index, f"ZeroShot_Score_{vlms.name}"] = zero_shot_result.get("Output_score", 0.0)
    df.at[index, f"Correct_{vlms.name}"] = 1 if predicted_label == row["Label"] else 0

df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"save {len(df)} list {output_csv} successful!")

true_labels = df["Label"].tolist()
predicted_labels = df[f"ZeroShot_Output_{vlms.name}"]
cm = confusion_matrix(true_labels, predicted_labels, labels=zero_shot_labels)
cm_df = pd.DataFrame(cm, index=zero_shot_labels, columns=zero_shot_labels)
cm_model_csv = output_csv.replace(".csv", f"_confusion_matrix_{vlms.name}.csv")
cm_df.to_csv(cm_model_csv)
print(f"Confusion Matrix ({vlms.name}) was save {cm_model_csv}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
plt.title(f"Confusion Matrix ({vlms.name})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

report = classification_report(true_labels, predicted_labels, target_names=zero_shot_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_csv = output_csv.replace(".csv", f"_classification_report_{vlms.name}.csv")
report_df.to_csv(report_csv)
print(f"Classification Report ({vlms.name}) was save {report_csv}")
