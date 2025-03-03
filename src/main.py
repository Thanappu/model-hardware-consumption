import sys
print(sys.path)
import src.model.vlms.vit_gpt2 as wedo_vit
import src.model.vlms.blip_large as wedo_blip
import src.model.vlms.blip_base as wedo_blip_base
import src.model.zeroshot_classif.deberta_v3_large as wedo_zeroshot
import  cv2

print(sys.path) #print path that system find package

vlms = wedo_vit.VlmsVit()
#vlms = wedo_blip.VlmsBlipLarge()
zero_shot = wedo_zeroshot.ZeroshotDebertaLarge()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    vlms_result = vlms.predict(frame)
    result = zero_shot.predict(vlms_result['response'])

    print(vlms_result)
    print(result['Output']," ",result['Output_score'])
