EPOCHS = 5
BATCH_SIZE = 16
DEVICE = 'cuda'
ROOT_PATH = '../input/camvid'

# the classes that we want to train
CLASSES_TO_TRAIN = ['animal', 'archway', 'bicyclist', 'bridge', 'building', 'car', 
        'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve', 
        'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving', 
        'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk', 
        'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight', 
        'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void',
        'wall']
# CLASSES_TO_TRAIN = ['lanemarkingdrve']

# DEBUG for visualizations
DEBUG = True