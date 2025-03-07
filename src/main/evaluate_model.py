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

dataset_path = '/Volumes/pond/model_evaluate/image_dataset/office_classification/test'
output_csv = f"/Users/thanapoompumee/Documents/project/file/file_evaluate_models/evaluation_results.csv"
confusion_matrix_csv = output_csv.replace(".csv", "_confusion_matrix.csv")

vlm_models = { "blip_base": wedo_blip_base.VlmsBlipBase(),
                "blip_large": wedo_blip_large.VlmsBlipLarge(),
                "vit_gpt2":wedo_vit.VlmsVit(),
}
zero_shot = {"DebertaLarge": wedo_zeroshot.ZeroshotDebertaLarge()
}
columns_data=["Image_Path", "Label"]
for vlm_model in vlm_models.keys():
    for zero_shot_model in zero_shot.keys():
        columns_data.append(f"{vlm_model}_{zero_shot_model}")
        columns_data.append(f"Score_{zero_shot_model}_{vlm_model}")
    columns_data.append(f"Correct_{vlm_model}")
df = pd.DataFrame(columns=columns_data)

data_rows = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                row_data = {"Image_Path": img_path,
                            "Label": class_name,
                }
                for model_name, model in vlm_models.items():
                    vlm_result = model.predict(img)
                    response = vlm_result['response']
                    for zero_shot_name, zero_shot_model in zero_shot.items():
                        result = zero_shot_model.predict(response)
                        row_data[f"{model_name}_{zero_shot_name}"]= result['Output']
                        row_data[f"Score_{zero_shot_name}_{model_name}"] = result['Output_score']
                    row_data[f"Correct_{model_name}"] = (result['Output'].strip().lower() == class_name.strip().lower())
                data_rows.append(row_data)

df = pd.DataFrame(data_rows, columns=columns_data)
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"save {len(df)} list {output_csv} successful!")

for vlm_name in vlm_models.keys():
    zero_shot_columns = [f"{vlm_name}_{zero_shot_name}" for zero_shot_name in zero_shot.keys()]
    predicted_labels = df[zero_shot_columns].mode(axis=1)[0]
    true_labels = df["Label"].tolist()

    zero_shot_labels = sorted(df["Label"].unique())

    cm = confusion_matrix(true_labels, predicted_labels, labels=zero_shot_labels)
    cm_df = pd.DataFrame(cm, index=zero_shot_labels, columns=zero_shot_labels)

    cm_model_csv = output_csv.replace(".csv", f"_confusion_matrix_{vlm_name}.csv")
    cm_df.to_csv(cm_model_csv)
    print(f"Confusion Matrix ({vlm_name}) was saved: {cm_model_csv}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
    plt.title(f"Confusion Matrix ({vlm_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    report = classification_report(true_labels, predicted_labels, target_names=zero_shot_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_csv = output_csv.replace(".csv", f"_classification_report_{vlm_name}.csv")
    report_df.to_csv(report_csv)
    print(f"Classification Report ({vlm_name}) was saved: {report_csv}")

