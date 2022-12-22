import cv2
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from util import utils
from util import image_process

sns.set()
params = {'legend.fontsize': 20,
          'legend.handlelength': 2,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'axes.titlesize': 20,
          'axes.labelsize': 16}
plt.rcParams.update(params)


def plot_color_bar_histogram(colors, frequencies):

    chart_height, chart_width = 50, 500
    chart = np.zeros((chart_height, chart_width, 3), np.uint8)
    hist = np.histogram(frequencies, bins=len(colors))
    start = 0
    slice_ = chart_width / len(frequencies)

    # creating color rectangles
    for i in range(len(colors)):
        end = start + slice_ * hist[0][i]

        b = int(colors[i][0])
        g = int(colors[i][1])
        r = int(colors[i][2])

        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
        start = end

    # display chart
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()


def plot_colors_in_colorspace(samples, color_model='bgr', marker_size=4):

    basic_colors = [
        np.array([0, 0, 255]),
        np.array([0, 255, 0]),
        np.array([255, 0, 0]),
        np.array([255, 255, 255]),
        np.array([0, 0, 0]),
    ]

    samples = np.concatenate((samples, basic_colors), axis=0)

    if color_model == 'hsv':
        colors = [image_process.bgr_to_hsv(b, g, r) for b, g, r in samples]
        xlabel = 'value'
        ylabel = 'saturation'
        zlabel = 'hue'
    elif color_model == 'spherical':
        colors = [image_process.bgr_to_spherical(b, g, r) for b, g, r in samples]
        colors = utils.normalize(colors, axis=0)
        xlabel = 'phi'
        ylabel = 'theta'
        zlabel = 'rho'
    else:
        colors = samples
        xlabel = 'red'
        ylabel = 'green'
        zlabel = 'blue'

    colors = np.array(colors)
    x_values = colors[:, 0]
    y_values = colors[:, 1]
    z_values = colors[:, 2]
    c_hex = np.array([image_process.bgr_to_hex(c) for c in samples])

    fig = go.Figure(data=go.Scatter3d(
        x=z_values, y=y_values, z=x_values,
        mode='markers',
        marker=dict(size=marker_size, color=c_hex, opacity=1),
        name="RGB Color Model"))

    fig.update_layout(scene=dict(
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        zaxis=dict(title=zlabel)))
    fig.show()


def plot_clustered_samples(samples, labels, color_labels, marker_size=4):
    basic_colors = np.array([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [255, 255, 255],
        [0, 0, 0],
    ])

    colors = np.array(samples)
    b_values = colors[:, 0]
    g_values = colors[:, 1]
    r_values = colors[:, 2]
    c_hex = np.array([image_process.bgr_to_hex(color_labels[label]) for label in labels])

    basic_b_values = basic_colors[:, 0]
    basic_g_values = basic_colors[:, 1]
    basic_r_values = basic_colors[:, 2]
    basic_c_hex = np.array([image_process.bgr_to_hex(c) for c in basic_colors])

    b_values = np.concatenate((b_values, basic_b_values), axis=0)
    g_values = np.concatenate((g_values, basic_g_values), axis=0)
    r_values = np.concatenate((r_values, basic_r_values), axis=0)
    c_hex = np.concatenate((c_hex, basic_c_hex), axis=0)

    fig = go.Figure(data=go.Scatter3d(
        x=r_values, y=g_values, z=b_values,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=c_hex,
            opacity=1),
        name="Labeled Samples in RGB"))

    fig.update_layout(scene=dict(
        xaxis=dict(
            title='red'),
        yaxis=dict(
            title='green'),
        zaxis=dict(
            title='blue')))
    fig.show()


def plot_siamese_results(history):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig, (loss_plot, distances, dist_ratio) = plt.subplots(1, 3, figsize=(10, 4))

    loss_plot.plot(epochs_range, history['train_loss'], color='red', label='train loss')
    loss_plot.set_title('Epochs - Loss')
    loss_plot.legend()

    distances.plot(epochs_range, history['positive_distance'], color='green', label='positive dist.')
    distances.plot(epochs_range, history['negative_distance'], color='red', label='negative dist.')
    distances.set_title('Epochs - Pos/Neg distance')
    distances.legend()

    dist_ratio.plot(epochs_range, history['distance_ratio'], color='red', label='distance ratio')
    dist_ratio.set_title('Epochs - Distance ratio')
    dist_ratio.legend()

    plt.show()


def plot_pix2pix_results(history):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig, (discriminator_loss, generator_loss) = plt.subplots(2, 1, figsize=(20, 8))

    discriminator_loss.plot(epochs_range, history['discriminator_real_loss'], color='green', label='Discriminator real loss')
    discriminator_loss.plot(epochs_range, history['discriminator_fake_loss'], color='red', label='Discriminator fake loss')
    discriminator_loss.set_title('Discriminator Real/Fake Loss')
    discriminator_loss.set_xlabel('Epochs')
    discriminator_loss.set_ylabel('Loss')
    discriminator_loss.legend()

    generator_loss.plot(epochs_range, history['generator_gan_loss'], color='red', label='Generator GAN loss')
    generator_loss.plot(epochs_range, history['generator_l1_loss'], color='green', label='Generator L1 loss')
    generator_loss.set_title('Generator GAN/L1 Loss')
    generator_loss.set_xlabel('Epochs')
    generator_loss.set_ylabel('Loss')
    generator_loss.legend()

    plt.show()


def plot_footandball_results(history):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig, ((total, ball_conf), (player_conf, player_loc)) = plt.subplots(2, 2, figsize=(20, 8))

    total.plot(epochs_range, history['train_loss'], color='green', label='Train Loss')
    total.plot(epochs_range, history['val_loss'], color='red', label='Valid. Loss')
    total.set_title('Total Loss')
    # total.set_xlabel('Epochs')
    total.set_ylabel('Loss')
    total.legend()

    ball_conf.plot(epochs_range, history['train_loss_ball_c'], color='green', label='Train Loss')
    ball_conf.plot(epochs_range, history['val_loss_ball_c'], color='red', label='Valid. Loss')
    ball_conf.set_title('Ball Confidence')
    # ball_conf.set_xlabel('Epochs')
    # ball_conf.set_ylabel('Loss')
    ball_conf.legend()

    player_conf.plot(epochs_range, history['train_loss_player_c'], color='green', label='Train Loss')
    player_conf.plot(epochs_range, history['val_loss_player_c'], color='red', label='Valid. Loss')
    player_conf.set_title('Player Confidence')
    player_conf.set_xlabel('Epochs')
    player_conf.set_ylabel('Loss')
    player_conf.legend()

    player_loc.plot(epochs_range, history['train_loss_player_l'], color='green', label='Train Loss')
    player_loc.plot(epochs_range, history['val_loss_player_l'], color='red', label='Valid. Loss')
    player_loc.set_title('Player Localization')
    player_loc.set_xlabel('Epochs')
    # player_loc.set_ylabel('Loss')
    player_loc.legend()

    plt.show()


def plot_footandball_results2(history):

    num_of_epochs = len(history[next(iter(history))])
    epochs_range = [i for i in range(num_of_epochs)]

    fig = make_subplots(rows=1, cols=2)

    train_plot_line = go.Scatter(x=epochs_range, y=history['train_loss'],
                                 marker=dict(
                                     size=2,
                                     color='red',
                                     opacity=1),
                                 name='Train Loss')
    val_plot_line = go.Scatter(x=epochs_range, y=history['val_loss'],
                               marker=dict(
                                   size=2,
                                   color='green',
                                   opacity=1),
                               name='Valid. Loss')
    fig.add_traces([train_plot_line, val_plot_line], 1, 1)
    fig['layout'].update(title='Total Loss')

    train_plot_line = go.Scatter(x=epochs_range, y=history['train_loss_ball_c'])
    val_plot_line = go.Scatter(x=epochs_range, y=history['val_loss_ball_c'])
    fig.add_traces([train_plot_line, val_plot_line], 1, 2)
    fig['layout'].update(title='Ball Confidence Loss')

    fig.show()

    # total.plot(epochs_range, history['train_loss'], color='green', label='Train Loss')
    # total.plot(epochs_range, history['val_loss'], color='red', label='Valid. Loss')
    # total.set_title('Total Loss')
    # # total.set_xlabel('Epochs')
    # total.set_ylabel('Loss')
    # total.legend()

    # ball_conf.plot(epochs_range, history['train_loss_ball_c'], color='green', label='Train Loss')
    # ball_conf.plot(epochs_range, history['val_loss_ball_c'], color='red', label='Valid. Loss')
    # ball_conf.set_title('Ball Confidence')
    # # ball_conf.set_xlabel('Epochs')
    # # ball_conf.set_ylabel('Loss')
    # ball_conf.legend()
    #
    # player_conf.plot(epochs_range, history['train_loss_player_c'], color='green', label='Train Loss')
    # player_conf.plot(epochs_range, history['val_loss_player_c'], color='red', label='Valid. Loss')
    # player_conf.set_title('Player Confidence')
    # player_conf.set_xlabel('Epochs')
    # player_conf.set_ylabel('Loss')
    # player_conf.legend()
    #
    # player_loc.plot(epochs_range, history['train_loss_player_l'], color='green', label='Train Loss')
    # player_loc.plot(epochs_range, history['val_loss_player_l'], color='red', label='Valid. Loss')
    # player_loc.set_title('Player Localization')
    # player_loc.set_xlabel('Epochs')
    # # player_loc.set_ylabel('Loss')
    # player_loc.legend()

    # plt.show()


def plot_logistic_regression_points(logreg, x, y):
    w, h = 920, 592
    x_min, x_max = 0, w
    y_min, y_max = 0, h
    h = 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    dpi = 80
    fig = plt.figure(figsize=(1200/dpi, 800/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.pcolormesh(xx, yy, Z, cmap=plt.get_cmap('coolwarm'))

    # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, s=128, edgecolors='k', cmap=plt.get_cmap('coolwarm'))
    # plt.xlabel('Court height')
    # plt.ylabel('Court width')

    # (920, 592)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plt.show()

    return plot2image(fig)


def plot2image(fig):
    fig.canvas.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


if __name__ == '__main__':
    import sys
    from models.foot_and_ball.network.footandball import FootAndBall

    filename = f'{utils.get_project_root()}tasks/results/PlayerBallDetector/trials_results/d_color_jitter_shrink_06-1/object_detector_10_b.pth'
    _, history = FootAndBall.load_model(filename, model=False, history=True)

    plot_footandball_results(history)

    sys.exit()