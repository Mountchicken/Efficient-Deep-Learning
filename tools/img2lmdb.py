"""a modified version of CRNN torch repository
https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py."""
import os

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(imagePath, annoPath, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    args:
        imagePath  : path to images
        annoPath     : path to annotations
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
    e.g.
    for text recognition task, the file structure is as follow:
      data
         |_image
               |_ img1.jpg
               |_ img2.jpg
         |_label
               |_ label1.txt
               |_ label2.txt
      label1.txt: 'img1.jpg Hello'
      label2.txt: 'img2.jpg World'
    the input params should be:
      imagePath='data/image'
      annoPath='data/label'
      outputPath='lmdbOut'
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=109951162776)
    cache = {}
    cnt = 1
    anno_list = os.listdir(annoPath)
    nSamples = len(anno_list)
    for anno in tqdm(anno_list):
        with open(os.path.join(annoPath, anno), 'r') as f:
            label = f.read().strip('\n')
        image_name, word = label.split[' ']
        image_path = os.path.join(imagePath, image_name)
        if not os.path.exists(image_path):
            print('%s does not exist' % image_path)
            continue
        with open(image_path, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except Exception:
                print('error occured', image_name)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{image_name} occurred error\n')
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    imagePath = ''
    annoPath = ''
    outputPath = ''
    createDataset(imagePath, annoPath, outputPath)
