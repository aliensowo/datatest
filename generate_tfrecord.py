"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Derbent':
        return 1
    elif row_label == 'Strigament_nastoyka':
        return 2
    elif row_label == 'Sangria':
        return 3
    elif row_label == 'Chocopie':
        return 4
    elif row_label == 'Cornline_bar':
        return 5
    elif row_label == 'Korovka_iz_korenovki_sguchonka':
        return 6
    elif row_label == 'Heinz_ketchup':
        return 7
    elif row_label == 'Geksoral_rastvor':
        return 8
    elif row_label == 'Ambrobene_rastvor':
        return 9
    elif row_label == 'Omeprazol_kapsuli':
        return 10
    elif row_label == 'Taustin_S':
        return 11
    elif row_label == 'Twix_solty_caramel':
        return 12
    elif row_label == 'Lowen_prise':
        return 13
    elif row_label == 'Lenta_salfetki_dlya_optiki':
        return 14
    elif row_label == 'Tigers':
        return 15
    elif row_label == 'Golden_virgina_orig':
        return 16
    elif row_label == 'Baisad_mayonez':
        return 17
    elif row_label == 'Bonfesto_cremolle':
        return 18
    elif row_label == 'Heinz_cheesy_sauce':
        return 19
    elif row_label == 'Ahmad_tea_earl_grey':
        return 20
    elif row_label == 'Hilltop_tea':
        return 21
    elif row_label == 'Fitotea_A':
        return 22
    elif row_label == 'Vitamin_AE_forte':
        return 23
    elif row_label == 'Al_Abbas_ceylon_tea':
        return 24
    elif row_label == 'Axa_Granola':
        return 25
    elif row_label == 'Muesli_Cranberry':
        return 26
    elif row_label == 'Nurofen_exressforte':
        return 27
    elif row_label == 'Prostamol_uno':
        return 28
    elif row_label == 'Afludol_detskiy':
        return 29
    elif row_label == 'Augmentin':
        return 30
    elif row_label == 'Pirantel':
        return 31
    elif row_label == 'Medisorb_Paracetamol':
        return 32
    elif row_label == 'Nimulid':
        return 33
    elif row_label == 'Diakarb':
        return 34
    elif row_label == 'Lowen_prise_gold':
        return 35
    elif row_label == 'Puzzle':
        return 36
    elif row_label == 'Adlin':
        return 37
    elif row_label == 'Koska_halva':
        return 38
    elif row_label == 'Koska_delight':
        return 39
    elif row_label == 'Sadi_pridoniya_sok_vishnya':
        return 40
    elif row_label == 'Sadi_pridoniya_sok_grusha':
        return 41
    elif row_label == 'Sadi_pridoniya_pure_apple':
        return 42
    elif row_label == 'Sanitaizer_zero':
        return 43
    elif row_label == 'Baryer':
        return 44
    elif row_label == 'Sadi_pridoniya_sok_abricos':
        return 45
    elif row_label == 'Sadi_pridoniya_spure_vishnya':
        return 46
    elif row_label == 'Muesli_ogo':
        return 47
    elif row_label == 'Sadaka_tea':
        return 48
    elif row_label == 'Barbaris':
        return 49
    elif row_label == 'Mens_peris':
        return 50
    elif row_label == 'Heinz_apple_pear':
        return 51
    elif row_label == 'Heinz_fruits_salad':
        return 52
    elif row_label == 'Heinz_fruits_pure':
        return 53
    elif row_label == 'Fruto_nyanya_apple_banana':
        return 54
    elif row_label == 'Heinz_kasha_before_sleep':
        return 55
    elif row_label == 'Nestle_ovsyanaya_kasha':
        return 56
    elif row_label == 'Plombir_fistashka':
        return 57
    elif row_label == 'Plombir_choco':
        return 58
    elif row_label == 'Givenchy_mascara':
        return 59
    elif row_label == 'Dom_v_kotorom':
        return 60
    elif row_label == 'Tide':
        return 61
    elif row_label == 'Five_lakes':
        return 62
    elif row_label == 'Kindzmarauli':
        return 63
    elif row_label == 'Vernel':
        return 64
    elif row_label == 'Barhatnie_ruchki':
        return 65
    elif row_label == 'Dolce_Albero_cake':
        return 66
    elif row_label == 'Voda_Agusha_5L':
        return 67
    elif row_label == 'Voda_Heney_kid_1L':
        return 68
    elif row_label == 'Sensiro_Suru_cream':
        return 69
    elif row_label == 'Greenfield':
        return 70
    elif row_label == 'Oil_of_black_Tmin':
        return 71
    elif row_label == 'Heinz_kasha_tykwa':
        return 72
    elif row_label == 'Risotto_porcino':
        return 73
    elif row_label == 'Tequila':
        return 74
    elif row_label == 'Mistal_ris_indika':
        return 75
    elif row_label == 'Mistal_grecha':
        return 76
    elif row_label == 'Mistral_grechka_farm':
        return 77
    elif row_label == 'Oil_amarant':
        return 78
    elif row_label == 'Oil_gretsky_orex':
        return 79
    elif row_label == 'Risotto_tartufo':
        return 80
    elif row_label == 'Richman_ceylon':
        return 81
    elif row_label == 'Perekrestok_malina':
        return 82
    elif row_label == 'Vologodskaya_malina':
        return 83
    elif row_label == 'Noweda':
        return 84
    elif row_label == 'Beaf_tushenka':
        return 85
    elif row_label == 'Beaf_extra':
        return 86
    elif row_label == 'Leovit_kisel':
        return 87
    elif row_label == 'Shipovnik':
        return 88
    elif row_label == 'Leps':
        return 89
    elif row_label == 'Voda_antipodes':
        return 90
    elif row_label == 'Barilla_pipe':
        return 91
    elif row_label == 'Batjeman_barton':
        return 92
    elif row_label == 'Sugar_melon':
        return 93
    elif row_label == 'Kalendula':
        return 94
    elif row_label == 'Bessmertnik':
        return 95
    elif row_label == 'Dammann':
        return 96
    elif row_label == 'Alfa_bank_Nikolaeva':
        return 97
    elif row_label == 'Alfa_bank_Sereda':
        return 98
    elif row_label == 'MKB_Pogidaev':
        return 99
    elif row_label == 'MKB_Filippenkov':
        return 100
    elif row_label == 'MiT_Losev':
        return 101
    elif row_label == 'MiT_Matnishyan':
        return 102
    elif row_label == 'Carmex':
        return 103
    elif row_label == 'Miratorg':
        return 104
    elif row_label == 'MKB_Logo':
        return 105
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
