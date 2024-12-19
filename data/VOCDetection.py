def resize_target(target, original_size, new_size):
    x_scale = new_size[0] / original_size[0]
    y_scale = new_size[1] / original_size[1]

    for obj in target['annotation']['object']:
        bbox = obj['bndbox']
        bbox['xmin'] = int(float(bbox['xmin']) * x_scale)
        bbox['ymin'] = int(float(bbox['ymin']) * y_scale)
        bbox['xmax'] = int(float(bbox['xmax']) * x_scale)
        bbox['ymax'] = int(float(bbox['ymax']) * y_scale)
    return target


def target_transform_func(target):
    original_size = (int(target['annotation']['size']['width']), int(target['annotation']['size']['height']))
    new_size = (224, 224)
    return resize_target(target, original_size, new_size)


def collate_fn(batch):
    return tuple(zip(*batch))