import sys
from models.two_pix2pix.test_options import opt
from models.two_pix2pix.data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.two_pix2pix.network.two_pix2pix_model import TwoPix2PixModel
from util import utils

opt = opt()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

data_loader = CustomDatasetDataLoader()
print(data_loader.name())
data_loader.initialize(opt)
dataset = data_loader.load_data()

model = TwoPix2PixModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()

    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visuals = model.get_current_visuals()
    output_image = utils.concatenate_multiple_images(
        visuals['real_A'],
        visuals['fake_B'],
        visuals['fake_C'],
        visuals['fake_D'],
        visuals['real_D']
        )
    utils.save_image(output_image, f'{utils.get_project_root()}tasks/results/two_pix2pix/court_real_fake_{i}.jpg')

print(f'Results from test images have been saved at directory: {utils.get_project_root()}tasks/results/two_pix2pix/')
sys.exit()
