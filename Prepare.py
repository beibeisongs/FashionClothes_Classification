# encoding=utf-8
# Date: 2018-10-14
# Author: MJUZY
# Important Reference: https://blog.csdn.net/JinbaoSite/article/details/77435558

# Attention: Part of the work for preparing has been down by the python scripts
#               which is located in D:\Dataset：DeepFashion\TrainDeepFashionModel_VGG16ed\Prepare.py
#
#               So please review it carefully !

# Attention: I have done some class combination operation, refering to the document
#               located in the directory of D:\Dataset_Clothes_Main_Small, please review it, too

# now start the preparation operation left for the Clothes_Classification project


import os
import shutil


original_dataset_dir = "D:/Dataset_Clothes_Main_Small"

base_dir = "D:\Dataset_Clothes_Main_Small\_ForModel"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

def make_the_dirs(base_dir, op):

    train_dir = os.path.join(base_dir, op)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    train_Blouse_dir = os.path.join(train_dir, "Blouse")
    if not os.path.exists(train_Blouse_dir):
        os.mkdir(train_Blouse_dir)
    train_Dress_dir = os.path.join(train_dir, "Dress")
    if not os.path.exists(train_Dress_dir):
        os.mkdir(train_Dress_dir)
    train_Jacket_dir = os.path.join(train_dir, "Jacket")
    if not os.path.exists(train_Jacket_dir):
        os.mkdir(train_Jacket_dir)
    train_Jeans_dir = os.path.join(train_dir, "Jeans")
    if not os.path.exists(train_Jeans_dir):
        os.mkdir(train_Jeans_dir)
    train_Shorts_dir = os.path.join(train_dir, "Shorts")
    if not os.path.exists(train_Shorts_dir):
        os.mkdir(train_Shorts_dir)
    train_Skirt_dir = os.path.join(train_dir, "Skirt")
    if not os.path.exists(train_Skirt_dir):
        os.mkdir(train_Skirt_dir)
    train_Sweater_dir = os.path.join(train_dir, "Sweater")
    if not os.path.exists(train_Sweater_dir):
        os.mkdir(train_Sweater_dir)
    train_Tee_dir = os.path.join(train_dir, "Tee")
    if not os.path.exists(train_Tee_dir):
        os.mkdir(train_Tee_dir)

def make_dirs():
    make_the_dirs(base_dir, "train")
    make_the_dirs(base_dir, "validation")
    make_the_dirs(base_dir, "test")

make_dirs()

def copy_data_TVT(class_name, base_dir, original_dataset_dir):
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")

    train_class_dir = os.path.join(train_dir, class_name)
    length = len(os.listdir(train_class_dir))

    line_i = 0
    fnames = [class_name + ".{}.jpg".format(i) for i in range(length, 2000)]
    for fname in fnames:
        line_i += 1
        print(line_i, fname)
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_class_dir, fname)
        shutil.copyfile(src, dst)

    print("total training" + class_name + "images : ", len(os.listdir(train_class_dir)))

    validation_class_dir = os.path.join(validation_dir, class_name)
    length = len(os.listdir(validation_class_dir))

    line_i = 0
    fnames = [class_name + ".{}.jpg".format(i) for i in range(length, 4000)]
    for fname in fnames:
        line_i += 1
        print(line_i, fname)
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_class_dir, fname)
        shutil.copyfile(src, dst)

    print("total validation" + class_name + "images : ", len(os.listdir(validation_class_dir)))

    test_class_dir = os.path.join(test_dir, class_name)
    length = len(os.listdir(test_class_dir))

    line_i = 0
    fnames = [class_name + ".{}.jpg".format(i) for i in range(length, 4000)]
    for fname in fnames:
        line_i += 1
        print(line_i, fname)
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_class_dir, fname)
        shutil.copyfile(src, dst)

    print("total test" + class_name + "images : ", len(os.listdir(test_class_dir)))


copy_data_TVT("Blouse", base_dir, original_dataset_dir + '/' + "Blouse")
copy_data_TVT("Dress", base_dir, original_dataset_dir + '/' + "Dress")
copy_data_TVT("Jacket", base_dir, original_dataset_dir + '/' + "Jacket")
copy_data_TVT("Jeans", base_dir, original_dataset_dir + '/' + "Jeans")
copy_data_TVT("Shorts", base_dir, original_dataset_dir + '/' + "Shorts")
copy_data_TVT("Skirt", base_dir, original_dataset_dir + '/' + "Skirt")
copy_data_TVT("Sweater", base_dir, original_dataset_dir + '/' + "Sweater")
copy_data_TVT("Tee", base_dir, original_dataset_dir + '/' + "Tee")


def getPicNum(target_path_dir):
    """

    :param target_path_dir: 'D:/Dataset_Clothes/Tee'
    :return:
    """
    pic_num = 0
    for dirpath, dirnames, filenames in os.walk(target_path_dir):
        pic_num = len(filenames)
    return pic_num


def combineTwoClass(original_dataset_dir, class1, class2_target):
    """

    :param original_dataset_dir: sample: D:\Dataset_Clothes_Main_Small
    :param class1: the pictures in class1 will be moved to the document of class2_target
    :param class2_target:
    :return:
    """
    one_time = True
    class1_dirpath = original_dataset_dir + '/' + class1
    for dirpath, dirnames, filenames in os.walk(class1_dirpath):
        if one_time:
            target_path_dir = original_dataset_dir + '/' + class2_target
            current_pic_num = getPicNum(target_path_dir)
            _i = 0 + current_pic_num

            print("_i : ", _i)
            for filename in filenames:
                _i += 1

                current_path = class1_dirpath + '/' + filename
                new_path = target_path_dir + '/' + class2_target + '.' + str(_i) + '.jpg'
                shutil.move(current_path, new_path)

            one_time = False
        else:
            break


"""
    >>>original_dataset_dir = "D:\Dataset_Clothes_Main_Small"
    >>>combineTwoClass(original_dataset_dir, "Coat", "Jacket")
    """

