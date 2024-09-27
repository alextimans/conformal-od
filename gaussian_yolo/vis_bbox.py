"""
This module contains a function to visualize bounding boxes with uncertainties.
This code is taken from: https://github.com/motokimura/PyTorch_Gaussian_YOLOv3
"""

import numpy as np
import matplotlib.pyplot as plt


def vis_bbox(img, bbox, label=None, score=None, label_names=None, instance_colors=None, 
    sigma=[], sigma_scale_img=(1.0, 1.0), sigma_scale_xy=3.0, sigma_scale_wh=3.0,
    show_inner_bound=False, alpha=1., linewidth=1., ax=None):
    """Visualize bounding boxes inside the image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        sigma (iterable of tuples): List of uncertainties with shape :math:`(R, 4)`.
             Each value indicates uncertainties of the xywh coordinates.
        sigma_scale_img (list of float): scaling factor to visualize xy uncertainties.
             This compensates image scaling each for x and y axis of sigmas.
        sigma_scale_xy (float): scaling factor to visualize xy uncertainties.
             This emphasizes the xy uncertainties to be visualized (default: 3-sigma).
        sigma_scale_wh (float): scaling factor to visualize wh uncertainties.
             This emphasizes the wh uncertainties to be visualized (default: 3-sigma).
        show_inner_bound (bool): True when inner bound of wh uncertainty should be drawn.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    from: https://github.com/chainer/chainercv
    """
             
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    if ax is None:
        fig = plt.figure()
        _, h, w = img.shape
        w_ = w / 60.0
        h_ = w_ * (h / w)
        fig.set_size_inches((w_, h_))
        ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img.transpose((1, 2, 0)).astype(np.uint8))
    ax.axis('off')
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        x, y = bb[1], bb[0]
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            (x, y), width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))
        
        if sigma:
            img_scale_xaxis, img_scale_yaxis = sigma_scale_img
            sx, sy, sw, sh = sigma[i]
            
            # wh uncertainties
            dw = width * np.power(sw, sigma_scale_wh) - width
            dh = height * np.power(sh, sigma_scale_wh) - height
            dw *= img_scale_xaxis
            dh *= img_scale_yaxis
            ax.add_patch(plt.Rectangle(
                (x - 0.5 * dw, y - 0.5 * dh), width + dw, height + dh, fill=False,
                edgecolor=color, linewidth=linewidth * 0.6, alpha=alpha, linestyle='--'))

            if show_inner_bound:
                dw = width / np.power(sw, sigma_scale_wh) - width
                dh = height / np.power(sh, sigma_scale_wh) - height
                dw *= img_scale_xaxis
                dh *= img_scale_yaxis
                ax.add_patch(plt.Rectangle(
                    (x - 0.5 * dw, y - 0.5 * dh), width + dw, height + dh, fill=False,
                    edgecolor=color, linewidth=linewidth * 0.6, alpha=alpha, linestyle='--'))
            
            # xy uncertainties
            sx *= img_scale_xaxis * sigma_scale_xy
            sy *= img_scale_yaxis * sigma_scale_xy
            cx, cy = x + width * 0.5, y + height * 0.5

            ax.add_line(plt.Line2D(
                (cx - sx, cx + sx), (cy, cy),
                color=color, linewidth=linewidth * 1.4, alpha=alpha))

            ax.add_line(plt.Line2D(
                (cx, cx), (cy - sy, cy + sy),
                color=color, linewidth=linewidth * 1.4, alpha=alpha))

            #ax.annotate('',  xy=(cx - sx, cy), xytext=(cx + sx, cy),
            #    arrowprops=dict(arrowstyle='|-|, widthA=0.25, widthB=0.25',
            #    facecolor=color, edgecolor=color, alpha=alpha))

            #ax.annotate('',  xy=(cx, cy - sy), xytext=(cx, cy + sy),
            #    arrowprops=dict(arrowstyle='|-|, widthA=0.25, widthB=0.25',
            #    facecolor=color, edgecolor=color, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(x, y,
                    ': '.join(caption),
                    fontsize=12,
                    color='black',
                    style='italic',
                    bbox={'facecolor': color, 'edgecolor': color, 'alpha': 1, 'pad': 0})
    return fig, ax
