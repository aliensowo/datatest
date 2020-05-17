######## Object Detection for Image #########
# 
# Author: Khai Do
# Date: 9/3/2019

## Some parts of the code is copied from Tensorflow object detection
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


# Import libraries
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

start_time = time.time()
# What model
directPath = os.getcwd()
print(directPath)
MODEL_NAME = os.path.join(directPath, 'trained-inference-graphs/output_inference_graph_v1.pb')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(directPath, 'training/label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 110


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection



path_test = r"C:\ObjDetectRCNN\ObjectDetection\testdata"
num_files = len([f for f in os.listdir(path_test)
                if os.path.isfile(os.path.join(path_test, f))])

#file log
file_log = open("log_file_data.txt", "w")
path = os.path.splitext(os.path.join(directPath, 'image.jpg'))[0]
test_path = os.listdir(path_test)

print(f"Start process for {num_files} files")
file_log.write(f"Total files are count: {num_files}\n")
i_count = 0
for img in test_path:
    if img.endswith(".jpg") or img.endswith(".JPG"):
        print(f"Process №{i_count} for {img}")
        IMG_SIZE = (1200,900)
        #imagefile = os.path.join(directPath, 'image.jpg') 
        imagefile = os.path.join(path_test, img)
        img_array = cv2.imread(imagefile)  
        #image_np = cv2.resize(img_array, IMG_SIZE)
        image_np = img_array
        frames = 0  
        start = time.time()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Extract image tensor
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    i_count += 1
                    if int(classes[0][0]) == 1:
                        class_name = 'Nestle_Ovsyanaya_kasha'
                    elif int(classes[0][0]) == 2:
                        class_name = 'Nestle_Risovaya_kasha'
                    elif int(classes[0][0]) == 3:
                        class_name = 'Nestle_Grechnevaya_kasha'
                    elif int(classes[0][0]) == 4:
                        class_name = 'Nestle_Grechnevaya_kasha_BM'
                    elif int(classes[0][0]) == 5:
                        class_name = 'Tema_Chicken_bee'
                    elif int(classes[0][0]) == 6:
                        class_name = 'Heinz_Apple_pure'
                    elif int(classes[0][0]) == 7:
                        class_name = 'Sadi_Pridoniya_pure_apple'
                    elif int(classes[0][0]) == 8:
                        class_name = 'Galavani_Kindzmarauli'
                    elif int(classes[0][0]) == 9:
                        class_name = 'Soda_pischevaya'
                    elif int(classes[0][0]) == 10:
                        class_name = 'Egoiste_cafe_noir'
                    elif int(classes[0][0]) == 11:
                        class_name = 'Nestle_Nesquik_kakao'
                    elif int(classes[0][0]) == 12:
                        class_name = 'Babaevsky_Vdohnoveniye'
                    elif int(classes[0][0]) == 13:
                        class_name = 'Hlebny_Spas_ovsyanoe'
                    elif int(classes[0][0]) == 14:
                        class_name = 'Happy_corn_pop_corn'
                    elif int(classes[0][0]) == 15:
                        class_name = 'Platinum_maslo_tmina'
                    elif int(classes[0][0]) == 16:
                        class_name = 'Nestle_Nesquik_choco'
                    elif int(classes[0][0]) == 17:
                        class_name = 'Myllyn_Paras_4_zernovaya_kasha'
                    elif int(classes[0][0]) == 18:
                        class_name = 'Babkini_semechki'
                    elif int(classes[0][0]) == 19:
                        class_name = 'Ice_Palace_Tiramisu'
                    elif int(classes[0][0]) == 20:
                        class_name = 'Lenta_Maslo_slivochnoe450'
                    elif int(classes[0][0]) == 21:
                        class_name = 'Nestle_Nesquik_kakao'
                    elif int(classes[0][0]) == 22:
                        class_name = 'Babaevsky_Vdohnoveniye'
                    elif int(classes[0][0]) == 23:
                        class_name = 'Hlebny_Spas_ovsyanoe'
                    elif int(classes[0][0]) == 24:
                        class_name = 'Happy_corn_pop_corn'
                    elif int(classes[0][0]) == 25:
                        class_name = 'Platinum_maslo_tmina'
                    elif int(classes[0][0]) == 26:
                        class_name = 'Nestle_Nesquik_choco'
                    elif int(classes[0][0]) == 27:
                        class_name = 'Myllyn_Paras_4_zernovaya_kasha'
                    elif int(classes[0][0]) == 28:
                        class_name = 'Babkini_semechki'
                    elif int(classes[0][0]) == 29:
                        class_name = 'Ice_Palace_Tiramisu'
                    elif int(classes[0][0]) == 30:
                        class_name = 'Lenta_Maslo_slivochnoe450'
                    elif int(classes[0][0]) == 31:
                        class_name = 'Ice_Palace_3Choc'
                    elif int(classes[0][0]) == 32:
                        class_name = 'Nevskaya_kosmetika_ushastiy_nyan_krem_milo'
                    elif int(classes[0][0]) == 33:
                        class_name = 'Nevskaya_kosmetika_ushastiy_nyan_gel'
                    elif int(classes[0][0]) == 34:
                        class_name = 'Givenchy_rumyana_04'
                    elif int(classes[0][0]) == 35:
                        class_name = 'Givenchy_rumyana_07'
                    elif int(classes[0][0]) == 36:
                        class_name = 'Base_universal_tonik'
                    elif int(classes[0][0]) == 37:
                        class_name = 'Neovita_eye_care'
                    elif int(classes[0][0]) == 38:
                        class_name = 'Neovita_de_make_up'
                    elif int(classes[0][0]) == 39:
                        class_name = 'Max_Factor_Pan_stick'
                    elif int(classes[0][0]) == 40:
                        class_name = 'Rocs_Delicate_whitening'
                    elif int(classes[0][0]) == 41:
                        class_name = 'Parliament_aqua_blue'
                    elif int(classes[0][0]) == 42:
                        class_name = 'Parliament_pline'
                    elif int(classes[0][0]) == 43:
                        class_name = 'Fine_life_ris_propareniy'
                    elif int(classes[0][0]) == 44:
                        class_name = 'Nemoloko_ovsyanoe_classic'
                    elif int(classes[0][0]) == 45:
                        class_name = 'Voda_senegskaya_gaz'
                    elif int(classes[0][0]) == 46:
                        class_name = 'Johnson_utenok'
                    elif int(classes[0][0]) == 47:
                        class_name = 'Toothpaste_zact_lion'
                    elif int(classes[0][0]) == 48:
                        class_name = 'Isme_rasyan'
                    elif int(classes[0][0]) == 49:
                        class_name = 'blendamed_3dwhite'
                    elif int(classes[0][0]) == 50:
                        class_name = 'Rochjana_Asiatic'
                    elif int(classes[0][0]) == 51:
                        class_name = 'Unic'
                    elif int(classes[0][0]) == 52:
                        class_name = 'Krasnaya_cena_spichki'
                    elif int(classes[0][0]) == 53:
                        class_name = 'Styx_basisol'
                    elif int(classes[0][0]) == 54:
                        class_name = 'Studio_up'
                    elif int(classes[0][0]) == 55:
                        class_name = 'Bioderma'
                    elif int(classes[0][0]) == 56:
                        class_name = 'Concept_blonde'
                    elif int(classes[0][0]) == 57:
                        class_name = 'H2O_infinity'
                    elif int(classes[0][0]) == 58:
                        class_name = 'Tropicana'
                    elif int(classes[0][0]) == 59:
                        class_name = 'Kotanyi_perechnaya_smes'
                    elif int(classes[0][0]) == 60:
                        class_name = 'Cherniu_gracon_chai'
                    elif int(classes[0][0]) == 61:
                        class_name = 'Ogilvi_reklama'
                    elif int(classes[0][0]) == 62:
                        class_name = 'Dikkens_lavka'
                    elif int(classes[0][0]) == 63:
                        class_name = 'Carebeau'
                    elif int(classes[0][0]) == 64:
                        class_name = 'Persic_krem'
                    elif int(classes[0][0]) == 65:
                        class_name = 'Chanel_le_weekend'
                    elif int(classes[0][0]) == 66:
                        class_name = '50_kopeek'
                    elif int(classes[0][0]) == 67:
                        class_name = 'Roobins'
                    elif int(classes[0][0]) == 68:
                        class_name = 'MaLone'
                    elif int(classes[0][0]) == 69:
                        class_name = 'Armani'
                    elif int(classes[0][0]) == 70:
                        class_name = 'AND'
                    elif int(classes[0][0]) == 71:
                        class_name = 'Inglot'
                    elif int(classes[0][0]) == 72:
                        class_name = 'Eveline'
                    elif int(classes[0][0]) == 73:
                        class_name = 'Panasonic'
                    elif int(classes[0][0]) == 74:
                        class_name = 'Lancome_Idole'
                    elif int(classes[0][0]) == 75:
                        class_name = 'Valet_chervi'
                    elif int(classes[0][0]) == 76:
                        class_name = 'Semerka_bubi'
                    elif int(classes[0][0]) == 77:
                        class_name = 'Karti_suvenir'
                    elif int(classes[0][0]) == 78:
                        class_name = 'kub'
                    elif int(classes[0][0]) == 79:
                        class_name = 'vip'
                    elif int(classes[0][0]) == 80:
                        class_name = 'Magnolia'
                    elif int(classes[0][0]) == 81:
                        class_name = 'Muz_silver'
                    elif int(classes[0][0]) == 82:
                        class_name = 'Karcher'
                    elif int(classes[0][0]) == 83:
                        class_name = 'Blacky_dress'
                    elif int(classes[0][0]) == 84:
                        class_name = 'Zdorov.ru'
                    elif int(classes[0][0]) == 85:
                        class_name = 'IQtoy'
                    elif int(classes[0][0]) == 86:
                        class_name = 'Vkusomaniya'
                    elif int(classes[0][0]) == 87:
                        class_name = 'Rendez_vous'
                    elif int(classes[0][0]) == 88:
                        class_name = 'Gantil'
                    elif int(classes[0][0]) == 89:
                        class_name = 'Kantata'
                    elif int(classes[0][0]) == 90:
                        class_name = 'Inventive_retail'
                    elif int(classes[0][0]) == 91:
                        class_name = 'Zewa_kids'
                    elif int(classes[0][0]) == 92:
                        class_name = 'babochka'
                    elif int(classes[0][0]) == 93:
                        class_name = 'Chitai_gorod'
                    elif int(classes[0][0]) == 94:
                        class_name = 'Chief'
                    elif int(classes[0][0]) == 95:
                        class_name = '500_rub'
                    elif int(classes[0][0]) == 96:
                        class_name = '200_rub'
                    elif int(classes[0][0]) == 97:
                        class_name = '1000_rub'
                    elif int(classes[0][0]) == 98:
                        class_name = '50_rub'
                    elif int(classes[0][0]) == 99:
                        class_name = 'Sberbank_MC'
                    elif int(classes[0][0]) == 100:
                        class_name = 'VTB_MIR'
                    elif int(classes[0][0]) == 101:
                        class_name = 'Senator'
                    elif int(classes[0][0]) == 102:
                        class_name = 'avion_spichki'
                    elif int(classes[0][0]) == 103:
                        class_name = 'Heinz_5zlakov'
                    elif int(classes[0][0]) == 104:
                        class_name = 'Greenfield_earl_grey'
                    elif int(classes[0][0]) == 105:
                        class_name = 'Heinz_grechnevaya_bm'
                    elif int(classes[0][0]) == 106:
                        class_name = 'Magnit'
                    elif int(classes[0][0]) == 107:
                        class_name = '10_rub'
                    elif int(classes[0][0]) == 108:
                        class_name = '2_rub'
                    elif int(classes[0][0]) == 109:
                        class_name = '100_rub'
                    elif int(classes[0][0]) == 110:
                        class_name= 'Russian_standart'

                    file_log.write(f"\n file_number: {i_count}, name: {img}, class: {class_name}, score: {(scores[0][0])*100}% \n") 
        print(f"End for {img}")
                # Display output
                #cv2.imshow('object detection', image_np)
                
                #if cv2.waitKey(0) & 0xFF == ord('q'):
                    #cv2.imwrite('output.png',image_np)
                    #cv2.destroyAllWindows()
                    
print(f"End process. All files in dir ({num_files})")
print("--- %s seconds ---" % (time.time() - start_time))
file_log.write(f"Time spend in seconds: {time.time() - start_time}")
file_log.close()

