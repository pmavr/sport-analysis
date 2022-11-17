import time
from models.two_pix2pix.train_options import opt
from models.two_pix2pix.data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.two_pix2pix.network.two_pix2pix_model import TwoPix2PixModel

from util import utils

if __name__ == '__main__':
    opt = opt()
    assert(opt.model == 'two_pix2pix')
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = TwoPix2PixModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    grass_hist = {
            'discriminator_real_loss': [],
            'discriminator_fake_loss': [],
            'generator_gan_loss': [],
            'generator_l1_loss': []}

    edge_hist = {
            'discriminator_real_loss': [],
            'discriminator_fake_loss': [],
            'generator_gan_loss': [],
            'generator_l1_loss': []}

    epoch_train_data = len(dataset)
    epoch_count = opt.niter + opt.niter_decay + 1
    for epoch in range(1, epoch_count):
        epoch_start_time = time.time()

        grass_epoch_discriminator_real_loss = 0.
        grass_epoch_discriminator_fake_loss = 0.
        grass_epoch_generator_gan_loss = 0.
        grass_epoch_generator_l1_loss = 0.

        edge_epoch_discriminator_real_loss = 0.
        edge_epoch_discriminator_fake_loss = 0.
        edge_epoch_generator_gan_loss = 0.
        edge_epoch_generator_l1_loss = 0.

        for data in dataset:

            model.set_input(data)
            model.optimize_parameters()

            grass_error, edge_error = model.get_current_errors()
            grass_epoch_discriminator_real_loss += grass_error['D_real'].item() / epoch_train_data
            grass_epoch_discriminator_fake_loss += grass_error['D_fake'].item() / epoch_train_data
            grass_epoch_generator_gan_loss += grass_error['G_GAN'].item() / epoch_train_data
            grass_epoch_generator_l1_loss += grass_error['G_L1'].item() / epoch_train_data

            edge_epoch_discriminator_real_loss += edge_error['D_real'].item() / epoch_train_data
            edge_epoch_discriminator_fake_loss += edge_error['D_fake'].item() / epoch_train_data
            edge_epoch_generator_gan_loss += edge_error['G_GAN'].item() / epoch_train_data
            edge_epoch_generator_l1_loss += edge_error['G_L1'].item() / epoch_train_data

        grass_hist['discriminator_real_loss'].append(grass_epoch_discriminator_real_loss)
        grass_hist['discriminator_fake_loss'].append(grass_epoch_discriminator_fake_loss)
        grass_hist['generator_gan_loss'].append(grass_epoch_generator_gan_loss)
        grass_hist['generator_l1_loss'].append(grass_epoch_generator_l1_loss)

        edge_hist['discriminator_real_loss'].append(edge_epoch_discriminator_real_loss)
        edge_hist['discriminator_fake_loss'].append(edge_epoch_discriminator_fake_loss)
        edge_hist['generator_gan_loss'].append(edge_epoch_generator_gan_loss)
        edge_hist['generator_l1_loss'].append(edge_epoch_generator_l1_loss)


        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save(epoch)
            utils.save_to_pickle_file(grass_hist, f'{utils.get_two_pix2pix_model_path()}grass_hist_{epoch}.pkl')
            utils.save_to_pickle_file(edge_hist, f'{utils.get_two_pix2pix_model_path()}edge_hist_{epoch}.pkl')

        print('\nGrass mask model:')
        print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1} "
              f"| Discr. Real Loss: {grass_hist['discriminator_real_loss'][-1]:.5f} "
              f"| Discr. Fake Loss: {grass_hist['discriminator_fake_loss'][-1]:.3f} "
              f"| GAN Loss: {grass_hist['generator_gan_loss'][-1]:.3f} "
              f"| L1 Loss: {grass_hist['generator_l1_loss'][-1]:.3f}")

        print('Edge map model:')
        print(f"Epoch {epoch}/{opt.niter + opt.niter_decay + 1} "
              f"| Discr. Real Loss: {edge_hist['discriminator_real_loss'][-1]:.5f} "
              f"| Discr. Fake Loss: {edge_hist['discriminator_fake_loss'][-1]:.3f} "
              f"| GAN Loss: {edge_hist['generator_gan_loss'][-1]:.3f} "
              f"| L1 Loss: {edge_hist['generator_l1_loss'][-1]:.3f}")

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, epoch_count, time.time() - epoch_start_time))
        model.update_learning_rate()


