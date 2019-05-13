from pycocotools.coco import COCO
import numpy as np
from utils import unfold_label, shuffle_data
from PIL import Image
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.Resize((72,72)),
    transforms.RandomHorizontalFlip(p=0.75),
    transforms.RandomCrop(72, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((72,72)),
    #transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_domain_name():
    return {'0': 'cifar100', '1': 'daimlerpedcls', '2': 'gtsrb', '3': 'omniglot', '4': 'svhn', '5': 'imagenet12',
     '6': 'aircraft', '7': 'dtd', '8': 'vgg-flowers', '9': 'ucf101'}

def get_data_folder():
    data_folder = 'data/VD/decathlon-1.0/annotations/'
    train_data = ['cifar100_train.json',
                  'daimlerpedcls_train.json',
                  'gtsrb_train.json',
                  'omniglot_train.json',
                  'svhn_train.json',
                  'imagenet12_train.json',
                  'aircraft_train.json',  # for test domains.
                  'dtd_train.json',
                  'vgg-flowers_train.json',
                  'ucf101_train.json']

    val_data = ['cifar100_val.json',
                'daimlerpedcls_val.json',
                'gtsrb_val.json',
                'omniglot_val.json',
                'svhn_val.json',
                'imagenet12_val.json',
                'aircraft_val.json',  # for test domains.
                'dtd_val.json',
                'vgg-flowers_val.json',
                'ucf101_val.json']

    test_data = ['cifar100_test_stripped.json',
                 'daimlerpedcls_test_stripped.json',
                 'gtsrb_test_stripped.json',
                 'omniglot_test_stripped.json',
                 'svhn_test_stripped.json',
                 'imagenet12_test_stripped.json',
                 'aircraft_test_stripped.json',  # for test domains.
                 'dtd_test_stripped.json',
                 'vgg-flowers_test_stripped.json',
                 'ucf101_test_stripped.json']

    return data_folder, train_data, val_data, test_data

class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, metatest, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path, metatest)
        self.load_data(b_unfold_label)

    def configuration(self, flags, stage, file_path, metatest):
        if metatest == False:
            self.batch_size = flags.batch_size
        if metatest == True:
            self.batch_size = flags.batch_size_metatest
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage
        self.shuffled = False

    def load_data(self, b_unfold_label):
        file_path = self.file_path
        coco = COCO(file_path)
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        #print('COCO categories: \n{}\n'.format(' '.join(nms)))
        self.num_classes = len(cats)
        images = []
        labels = []
        for cat in cats:
            catIds = coco.getCatIds(catNms=cat['name'])
            # print(catIds)
            imgIds = coco.getImgIds(catIds=catIds)
            img = coco.loadImgs(imgIds)
            labels.extend([(catIds[0] % 10000 - 1) for i in range(len(img))])
            images.extend(img)
        if len(images) == 0:
            images = coco.dataset['images']
        if b_unfold_label:
            labels = unfold_label(labels=labels, classes=len(np.unique(labels)))
        #assert len(images) == len(labels)
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)
        if self.stage is 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    def get_images_labels_batch(self,batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1
            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train
                self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
            #img = cv2.imread(self.images[self.current_index]['file_name'])
            img = Image.open(self.images[self.current_index]['file_name'])
            img = img.convert('RGB')
            img = transform_train(img)
            img = np.array(img)
            images.append(img)
            labels.append(self.labels[self.current_index])

        return np.array(images), np.array(labels)

def get_image(images):
    images_data = []
    for img in images:
        img = Image.open(img['file_name'])
        img = img.convert('RGB')

        img = transform_test(img)
        img = np.array(img)
        images_data.append(img)

    return np.array(images_data)


