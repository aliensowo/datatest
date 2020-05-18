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



path_test = r"/home/user/Documents/ObjectDetection/testdata"
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
                        class_name = 'Derbent'
                    elif int(classes[0][0]) == 2:
                        class_name = 'Strigament_nastoyka'
                    elif int(classes[0][0]) == 3:
                        class_name = 'Sangria'
                    elif int(classes[0][0]) == 4:
                        class_name = 'Chocopie'
                    elif int(classes[0][0]) == 5:
                        class_name = 'Cornline_bar'
                    elif int(classes[0][0]) == 6:
                        class_name = 'Korovka_iz_korenovki_sguchonka'
                    elif int(classes[0][0]) == 7:
                        class_name = 'Heinz_ketchup'
                    elif int(classes[0][0]) == 8:
                        class_name = 'Geksoral_rastvor'
                    elif int(classes[0][0]) == 9:
                        class_name = 'Ambrobene_rastvor'
                    elif int(classes[0][0]) == 10:
                        class_name = 'Omeprazol_kapsuli'
                    elif int(classes[0][0]) == 11:
                        class_name = 'Taustin_S'
                    elif int(classes[0][0]) == 12:
                        class_name = 'Twix_solty_caramel'
                    elif int(classes[0][0]) == 13:
                        class_name = 'Lowen_prise'
                    elif int(classes[0][0]) == 14:
                        class_name = 'Lenta_salfetki_dlya_optiki'
                    elif int(classes[0][0]) == 15:
                        class_name = 'Tigers'
                    elif int(classes[0][0]) == 16:
                        class_name = 'Golden_virgina_orig'
                    elif int(classes[0][0]) == 17:
                        class_name = 'Baisad_mayonez'
                    elif int(classes[0][0]) == 18:
                        class_name = 'Bonfesto_cremolle'
                    elif int(classes[0][0]) == 19:
                        class_name = 'Heinz_cheesy_sauce'
                    elif int(classes[0][0]) == 20:
                        class_name = 'Ahmad_tea_earl_grey'
                    elif int(classes[0][0]) == 21:
                        class_name = 'Hilltop_tea'
                    elif int(classes[0][0]) == 22:
                        class_name = 'Fitotea_A'
                    elif int(classes[0][0]) == 23:
                        class_name = 'Vitamin_AE_forte'
                    elif int(classes[0][0]) == 24:
                        class_name = 'Al_Abbas_ceylon_tea'
                    elif int(classes[0][0]) == 25:
                        class_name = 'Axa_Granola'
                    elif int(classes[0][0]) == 26:
                        class_name = 'Muesli_Cranberry'
                    elif int(classes[0][0]) == 27:
                        class_name = 'Nurofen_exressforte'
                    elif int(classes[0][0]) == 28:
                        class_name = 'Prostamol_uno'
                    elif int(classes[0][0]) == 29:
                        class_name = 'Afludol_detskiy'
                    elif int(classes[0][0]) == 30:
                        class_name = 'Augmentin'
                    elif int(classes[0][0]) == 31:
                        class_name = 'Pirantel'
                    elif int(classes[0][0]) == 32:
                        class_name = 'Medisorb_Paracetamol'
                    elif int(classes[0][0]) == 33:
                        class_name = 'Nimulid'
                    elif int(classes[0][0]) == 34:
                        class_name = 'Diakarb'
                    elif int(classes[0][0]) == 35:
                        class_name = 'Lowen_prise_gold'
                    elif int(classes[0][0]) == 36:
                        class_name = 'Puzzle'
                    elif int(classes[0][0]) == 37:
                        class_name = 'Adlin'
                    elif int(classes[0][0]) == 38:
                        class_name = 'Koska_halva'
                    elif int(classes[0][0]) == 39:
                        class_name = 'Koska_delight'
                    elif int(classes[0][0]) == 40:
                        class_name = 'Sadi_pridoniya_sok_vishnya'
                    elif int(classes[0][0]) == 41:
                        class_name = 'Sadi_pridoniya_sok_grusha'
                    elif int(classes[0][0]) == 42:
                        class_name = 'Sadi_pridoniya_pure_apple'
                    elif int(classes[0][0]) == 43:
                        class_name = 'Sanitaizer_zero'
                    elif int(classes[0][0]) == 44:
                        class_name = 'Baryer'
                    elif int(classes[0][0]) == 45:
                        class_name = 'Sadi_pridoniya_sok_abricos'
                    elif int(classes[0][0]) == 46:
                        class_name = 'Sadi_pridoniya_spure_vishnya'
                    elif int(classes[0][0]) == 47:
                        class_name = 'Muesli_ogo'
                    elif int(classes[0][0]) == 48:
                        class_name = 'Sadaka_tea'
                    elif int(classes[0][0]) == 49:
                        class_name = 'Barbaris'
                    elif int(classes[0][0]) == 50:
                        class_name = 'Mens_peris'
                    elif int(classes[0][0]) == 51:
                        class_name = 'Heinz_apple_pear'
                    elif int(classes[0][0]) == 52:
                        class_name = 'Heinz_fruits_salad'
                    elif int(classes[0][0]) == 53:
                        class_name = 'Heinz_fruits_pure'
                    elif int(classes[0][0]) == 54:
                        class_name = 'Fruto_nyanya_apple_banana'
                    elif int(classes[0][0]) == 55:
                        class_name = 'Heinz_kasha_before_sleep'
                    elif int(classes[0][0]) == 56:
                        class_name = 'Nestle_ovsyanaya_kasha'
                    elif int(classes[0][0]) == 57:
                        class_name = 'Plombir_fistashka'
                    elif int(classes[0][0]) == 58:
                        class_name = 'Plombir_choco'
                    elif int(classes[0][0]) == 59:
                        class_name = 'Givenchy_mascara'
                    elif int(classes[0][0]) == 60:
                        class_name = 'Dom_v_kotorom'
                    elif int(classes[0][0]) == 61:
                        class_name = 'Tide'
                    elif int(classes[0][0]) == 62:
                        class_name = 'Five_lakes'
                    elif int(classes[0][0]) == 63:
                        class_name = 'Kindzmarauli'
                    elif int(classes[0][0]) == 64:
                        class_name = 'Vernel'
                    elif int(classes[0][0]) == 65:
                        class_name = 'Barhatnie_ruchki'
                    elif int(classes[0][0]) == 66:
                        class_name = 'Dolce_Albero_cake'
                    elif int(classes[0][0]) == 67:
                        class_name = 'Voda_Agusha_5L'
                    elif int(classes[0][0]) == 68:
                        class_name = 'Voda_Heney_kid_1L'
                    elif int(classes[0][0]) == 69:
                        class_name = 'Sensiro_Suru_cream'
                    elif int(classes[0][0]) == 70:
                        class_name = 'Greenfield'
                    elif int(classes[0][0]) == 71:
                        class_name = 'Oil_of_black_Tmin'
                    elif int(classes[0][0]) == 72:
                        class_name = 'Heinz_kasha_tykwa'
                    elif int(classes[0][0]) == 73:
                        class_name = 'Risotto_porcino'
                    elif int(classes[0][0]) == 74:
                        class_name = 'Tequila'
                    elif int(classes[0][0]) == 75:
                        class_name = 'Mistal_ris_indika'
                    elif int(classes[0][0]) == 76:
                        class_name = 'Mistal_grecha'
                    elif int(classes[0][0]) == 77:
                        class_name = 'Mistral_grechka_farm'
                    elif int(classes[0][0]) == 78:
                        class_name = 'Oil_amarant'
                    elif int(classes[0][0]) == 79:
                        class_name = 'Oil_gretsky_orex'
                    elif int(classes[0][0]) == 80:
                        class_name = 'Risotto_tartufo'
                    elif int(classes[0][0]) == 81:
                        class_name = 'Richman_ceylon'
                    elif int(classes[0][0]) == 82:
                        class_name = 'Perekrestok_malina'
                    elif int(classes[0][0]) == 83:
                        class_name = 'Vologodskaya_malina'
                    elif int(classes[0][0]) == 84:
                        class_name = 'Noweda'
                    elif int(classes[0][0]) == 85:
                        class_name = 'Beaf_tushenka'
                    elif int(classes[0][0]) == 86:
                        class_name = 'Beaf_extra'
                    elif int(classes[0][0]) == 87:
                        class_name = 'Leovit_kisel'
                    elif int(classes[0][0]) == 88:
                        class_name = 'Shipovnik'
                    elif int(classes[0][0]) == 89:
                        class_name = 'Leps'
                    elif int(classes[0][0]) == 90:
                        class_name = 'Voda_antipodes'
                    elif int(classes[0][0]) == 91:
                        class_name = 'Barilla_pipe'
                    elif int(classes[0][0]) == 92:
                        class_name = 'Batjeman_barton'
                    elif int(classes[0][0]) == 93:
                        class_name = 'Sugar_melon'
                    elif int(classes[0][0]) == 94:
                        class_name = 'Kalendula'
                    elif int(classes[0][0]) == 95:
                        class_name = 'Bessmertnik'
                    elif int(classes[0][0]) == 96:
                        class_name = 'Dammann'
                    elif int(classes[0][0]) == 97:
                        class_name = 'Alfa_bank_Nikolaeva'
                    elif int(classes[0][0]) == 98:
                        class_name = 'Alfa_bank_Sereda'
                    elif int(classes[0][0]) == 99:
                        class_name = 'MKB_Pogidaev'
                    elif int(classes[0][0]) == 100:
                        class_name = 'MKB_Filippenkov'
                    elif int(classes[0][0]) == 101:
                        class_name = 'MiT_Losev'
                    elif int(classes[0][0]) == 102:
                        class_name = 'MiT_Matnishyan'
                    elif int(classes[0][0]) == 103:
                        class_name = 'Carmex'
                    elif int(classes[0][0]) == 104:
                        class_name = 'Miratorg'
                    elif int(classes[0][0]) == 105:
                        class_name = 'MKB_Logo'


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

