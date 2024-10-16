
def min_max(data):
    _min, _max = data.min(), data.max()
    return (data - _min) / (_max - _min)


def mean_std(data):
    return (data - data.mean()) / data.std()


class ZScoreNormalization:
    @staticmethod
    def run(image, seg):
        mask = seg >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (max(std, 1e-8))

        return image
