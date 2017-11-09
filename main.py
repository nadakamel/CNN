import os, shutil, glob, random, time, keras
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau  
from keras.applications import Xception, InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Nadam, Adam, RMSprop, SGD
from keras.utils import np_utils
from PIL import Image

def get_session(gpu_fraction=0.3):
    #Assume that you have 12GB of GPU memory and want to allocate ~4GB

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#KTF.set_session(get_session())

FOLDERS_NAMES = ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove', '006.basketball-hoop', '007.bat', '008.bathtub', '009.bear', '010.beer-mug', '011.billiards', '012.binoculars', '013.birdbath', '014.blimp', '015.bonsai-101', '016.boom-box', '017.bowling-ball', '018.bowling-pin', '019.boxing-glove', '020.brain-101', '021.breadmaker', '022.buddha-101', '023.bulldozer', '024.butterfly', '025.cactus', '026.cake', '027.calculator', '028.camel', '029.cannon', '030.canoe', '031.car-tire', '032.cartman', '033.cd', '034.centipede', '035.cereal-box', '036.chandelier-101', '037.chess-board', '038.chimp', '039.chopsticks', '040.cockroach', '041.coffee-mug', '042.coffin', '043.coin', '044.comet', '045.computer-keyboard', '046.computer-monitor', '047.computer-mouse', '048.conch', '049.cormorant', '050.covered-wagon', '051.cowboy-hat', '052.crab-101', '053.desk-globe', '054.diamond-ring', '055.dice', '056.dog', '057.dolphin-101', '058.doorknob', '059.drinking-straw', '060.duck', '061.dumb-bell', '062.eiffel-tower', '063.electric-guitar-101', '064.elephant-101', '065.elk', '066.ewer-101', '067.eyeglasses', '068.fern', '069.fighter-jet', '070.fire-extinguisher', '071.fire-hydrant', '072.fire-truck', '073.fireworks', '074.flashlight', '075.floppy-disk', '076.football-helmet', '077.french-horn', '078.fried-egg', '079.frisbee', '080.frog', '081.frying-pan', '082.galaxy', '083.gas-pump', '084.giraffe', '085.goat', '086.golden-gate-bridge', '087.goldfish', '088.golf-ball', '089.goose', '090.gorilla', '091.grand-piano-101', '092.grapes', '093.grasshopper', '094.guitar-pick', '095.hamburger', '096.hammock', '097.harmonica', '098.harp', '099.harpsichord', '100.hawksbill-101', '101.head-phones', '102.helicopter-101', '103.hibiscus', '104.homer-simpson', '105.horse', '106.horseshoe-crab', '107.hot-air-balloon', '108.hot-dog', '109.hot-tub', '110.hourglass', '111.house-fly', '112.human-skeleton', '113.hummingbird', '114.ibis-101', '115.ice-cream-cone', '116.iguana', '117.ipod', '118.iris', '119.jesus-christ', '120.joy-stick', '121.kangaroo-101', '122.kayak', '123.ketch-101', '124.killer-whale', '125.knife', '126.ladder', '127.laptop-101', '128.lathe', '129.leopards-101', '130.license-plate', '131.lightbulb', '132.light-house', '133.lightning', '134.llama-101', '135.mailbox', '136.mandolin', '137.mars', '138.mattress', '139.megaphone', '140.menorah-101', '141.microscope', '142.microwave', '143.minaret', '144.minotaur', '145.motorbikes-101', '146.mountain-bike', '147.mushroom', '148.mussels', '149.necktie', '150.octopus', '151.ostrich', '152.owl', '153.palm-pilot', '154.palm-tree', '155.paperclip', '156.paper-shredder', '157.pci-card', '158.penguin', '159.people', '160.pez-dispenser', '161.photocopier', '162.picnic-table', '163.playing-card', '164.porcupine', '165.pram', '166.praying-mantis', '167.pyramid', '168.raccoon', '169.radio-telescope', '170.rainbow', '171.refrigerator', '172.revolver-101', '173.rifle', '174.rotary-phone', '175.roulette-wheel', '176.saddle', '177.saturn', '178.school-bus', '179.scorpion-101', '180.screwdriver', '181.segway', '182.self-propelled-lawn-mower', '183.sextant', '184.sheet-music', '185.skateboard', '186.skunk', '187.skyscraper', '188.smokestack', '189.snail', '190.snake', '191.sneaker', '192.snowmobile', '193.soccer-ball', '194.socks', '195.soda-can', '196.spaghetti', '197.speed-boat', '198.spider', '199.spoon', '200.stained-glass', '201.starfish-101', '202.steering-wheel', '203.stirrups', '204.sunflower-101', '205.superman', '206.sushi', '207.swan', '208.swiss-army-knife', '209.sword', '210.syringe', '211.tambourine', '212.teapot', '213.teddy-bear', '214.teepee', '215.telephone-box', '216.tennis-ball', '217.tennis-court', '218.tennis-racket', '219.theodolite', '220.toaster', '221.tomato', '222.tombstone', '223.top-hat', '224.touring-bike', '225.tower-pisa', '226.traffic-light', '227.treadmill', '228.triceratops', '229.tricycle', '230.trilobite-101', '231.tripod', '232.t-shirt', '233.tuning-fork', '234.tweezer', '235.umbrella-101', '236.unicorn', '237.vcr', '238.video-projector', '239.washing-machine', '240.watch-101', '241.waterfall', '242.watermelon', '243.welding-mask', '244.wheelbarrow', '245.windmill', '246.wine-bottle', '247.xylophone', '248.yarmulke', '249.yo-yo', '250.zebra', '251.airplanes-101', '252.car-side-101', '253.faces-easy-101', '254.greyhound', '255.tennis-shoes', '256.toad', '257.clutter']
TRAINING_DIR_PATH = './TrainingSet/'
VALIDATION_DIR_PATH = './ValidationSet/'

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
NUMBER_OF_CLASSES = 257

LEARNING_RATE = 0.001
MOMENTUM = 0.8
EPOCHS = 50
BATCH_SIZE = 8
NUM_OF_TRAINING_IMAGES = 26752
NUM_OF_VALIDATION_IMAGES = 3855

# Convert a tuple or struct_time representing a time as returned by gmtime() or localtime() in a date and time format
CURRENT_TIME = time.strftime("%c")
# Generating graphs from the TensorFlow processing at different paths for each run (based on time of run)
TENSORBOARD_CALLBACK = TensorBoard(log_dir='./logs/' + CURRENT_TIME, histogram_freq = 0, write_graph = True, write_images = False)


def resizing_dataset_size(width, height, folders_names):
    for category in folders_names:
        for filename in glob.iglob(TRAINING_DIR_PATH + category + '/*.jpg'):
            img = Image.open(filename)
            img = img.resize((width, height), Image.ANTIALIAS)
            img.save(filename)

def generate_validation_dataset(num_of_images_to_select):
    for category in FOLDERS_NAMES:
        create_folder(VALIDATION_DIR_PATH + category + '/')
        group_of_items = []
        for filename in glob.iglob(TRAINING_DIR_PATH + category + '/*.jpg'):
            group_of_items.append(filename)
        list_of_random_items = random.sample(group_of_items, num_of_images_to_select)
        for path in list_of_random_items:
            print path
            shutil.move(path, VALIDATION_DIR_PATH + category + '/')

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


tr_datagen = image.ImageDataGenerator(
    featurewise_center = False,             # Boolean. Set input mean to 0 over the dataset, feature-wise
    samplewise_center = False,              # Boolean. Set each sample mean to 0
    featurewise_std_normalization = False,  # Boolean. Divide inputs by std of the dataset, feature-wise
    samplewise_std_normalization = False,   # Boolean. Divide each input by its std
    zca_whitening = False,                  # Boolean. Apply ZCA whitening
    rotation_range = 15,                    # Int. Degree range for random rotations
    width_shift_range = 0.2,                # Float. Range for random horizontal shifts
    height_shift_range = 0.2,               # Float. Range for random vertical shifts
    shear_range = 0.2,                      # Float. Shear Intensity
    zoom_range = 0.2,                       # Float. Range for random zoom
    fill_mode = 'nearest',                  # Points outside the boundaries of the input are filled according to the default nearest state
    horizontal_flip = True,                 # Boolean. Randomly flip inputs horizontally
    vertical_flip = False)                  # Boolean. Randomly flip inputs vertically

# this is a generator that will read pictures found in
# subfoler 'TrainingSet', and indefinitely generate
# batches of augmented image data
tr_generator = tr_datagen.flow_from_directory(
        TRAINING_DIR_PATH,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')  # since we use categorical_crossentropy loss

val_datagen = image.ImageDataGenerator(rescale = None)

# this is a similar generator, for validation data in subfolder 'ValidationSet' 
val_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR_PATH,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

print("Dataset loaded!")

def train_model():
    # create the base pre-trained model
    base_model = Xception(weights='imagenet', include_top=False)
    #base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    #x = Dense(1024, activation='relu')(x)

    # and a logistic layer -- let's say we have 257 classes
    output_layer = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=output_layer)

    # saving the best weights found at current run of the network model, saved to binary file assigned to filepath
    checkpointer = ModelCheckpoint(filepath = "./weights.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')    # Reference: https://keras.io/callbacks/#modelcheckpoint

    # decaying learning rate when a plateau is reached in the validation loss by a factor of 0.2; reference: https://keras.io/callbacks/#reducelronplateau
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = 1e-6)

    # decide the optimizer
    optimizer = Nadam(lr = LEARNING_RATE)

    #model.load_weights('./weights.hdf5')        # Loading initial weights from file; reference: https://keras.io/getting-started/faq/

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])   # categorical_crossentropy is for negative log likelihood

    # print out summary of my model
    model.summary()      

    print 'Now Training...'
    # train model
    model.fit_generator(
            tr_generator,
            steps_per_epoch=NUM_OF_TRAINING_IMAGES//BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=NUM_OF_VALIDATION_IMAGES//BATCH_SIZE,
            verbose=1, callbacks=[TENSORBOARD_CALLBACK, checkpointer,reduce_lr])

#def test_model():


def main():
    #generate_validation_dataset(15)
    train_model()
    print 'Done!'

if __name__ == "__main__":
    main()