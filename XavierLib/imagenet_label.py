
import os
import numpy as np
import random
from PIL import Image


class wnid_to_2015(object):
    def __init__(self, filename):
        self.dict = {'background': 0}
        self.syns_list = ['background']
        line = 1
        for s in open(filename).readlines():
            wnid = s.strip()
            self.syns_list.append(wnid)
            self.dict[wnid] = line
            line += 1

    def get_num_label(self, wnid):
        return self.dict[wnid]

    def get_wnid_label(self, num):
        return self.syns_list[num]


def read_image(images_folder):
    image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
    im_array = Image.open(image_path).convert('RGB')
    return im_array

def one_hot(index):
    onehot = np.zeros(1001)
    onehot[index] = 1.0
    return onehot

def read_one(images_source, cls_2015):

    class_index = random.randint(1, 1000)

    folder = cls_2015.get_wnid_label(class_index)
    image = read_image(os.path.join(images_source, folder))
    label = one_hot(class_index)

    return image, label

def read_batch(batch_size, images_source, cls_2015):
    batch_images = []
    batch_labels = []

    for i in range(batch_size):
        class_index = random.randint(1, 1000)

        folder = cls_2015.get_wnid_label(class_index)
        batch_images.append(read_image(os.path.join(images_source, folder)))
        batch_labels.append(one_hot(class_index))

    np.vstack(batch_images)
    np.vstack(batch_labels)
    return batch_images, batch_labels


def preprocessing(img_path, nothing=True):
    if nothing:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float32)
        img = img / 255
    else:
        img = Image.open(img_path).convert('RGB')
        rate = 0.03235
        img = img.crop((int(img.size[0]*rate), int(img.size[1]*rate),
                        int(img.size[0]*(1-rate)), int(img.size[1]*(1-rate))))
        img = img.resize((224, 224), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img = img / 255
        img = 2*(img-0.5)
    return img


class validation_set_reader(object):
    def __init__(self, validation_source, annotations, cls_2015):
        self.remain_list = random.sample(range(50000), 50000)
        self.images_val = sorted(os.listdir(validation_source))
        self.validation_source = validation_source
        with open(annotations) as f:
            self.gt_idxs = []
            wnids = f.readlines()
            for wnid in wnids:
                self.gt_idxs.append(cls_2015.get_num_label(wnid.strip()))


    def read_validation_one(self):
        if len(self.remain_list) == 0:
            print('Shuffle...')
            self.remain_list = random.sample(range(50000), 50000)

        idx = self.remain_list[0]
        del self.remain_list[0]

        image = self.images_val[idx]
        images_val = preprocessing(os.path.join(self.validation_source, image))
        labels_val = one_hot(self.gt_idxs[idx])

        return images_val, labels_val


if __name__ == '__main__':
    converter = wnid_to_2015('E:/train_data/ImageNet/data/imagenet_lsvrc_2015_synsets.txt')
    print(converter.get_num_label('n01440764'), converter.get_wnid_label(converter.get_num_label('n01440764')))