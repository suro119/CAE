from collections import Counter
import pickle
import numpy as np
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq

import os

class RangeCoder():
    def __init__(self, histogram_path, codes_dir):
        self.histogram_path = histogram_path
        self.codes_dir = codes_dir
        self.histogram_dict = None
        self.cum_freq = None
        self.resolution = 16384

    def encode(self, code, test_num):
        encoder = RangeEncoder(os.path.join(self.codes_dir, str(test_num) + '.bin'))
        encoder.encode(code, self.cum_freq)
        encoder.close()

    def decode(self, length, test_num):
        decoder = RangeDecoder(os.path.join(self.codes_dir, str(test_num) + '.bin'))
        data = decoder.decode(length, self.cum_freq)
        data = np.array(data)
        decoder.close()
        return data

    def get_bpp(self, test_num, image_dim=(128,128)):
        # Returns the bpp (bits per pixel) of the 'test_num'th test image
        filename = os.path.join(self.codes_dir, str(test_num) + '.bin')
        num_bytes = os.path.getsize(filename)
        bpp = num_bytes * 8 / (image_dim[0] * image_dim[1])
        return bpp

    def get_histogram(self, train_dataset, model, batch_size=32):
        ''' 
        Gets the histogram of code coefficients
        Returns a dict with keys 'min', 'max', and 'list', where 'min/max' is the 
        min/max code values and 'list' is the histogram list
        '''
        try:
            with open(self.histogram_path, 'rb') as f:
                self.histogram_dict = pickle.load(f)
            print('Loaded histogram')
        except:
            print('Creating histrogram (will take 19200 iterations)')
            code_counter = Counter()
            model.eval()
            for i, data in enumerate(train_dataset):
                if i > 600:  # use 19200 images to calculate histogram
                    break
                model.set_input(data)
                model.test()
                code = model.code.reshape(-1).cpu().numpy()  # B x 96 x 16 x 16
                code_counter.update(code)

                if i % 20 == 0:
                    print('Iteration: {}'.format(i*batch_size))

            min_code = int(min(code_counter.keys()))
            max_code = int(max(code_counter.keys()))

            print('Min: {}, Max: {}'.format(min_code, max_code))

            self.histogram_dict = {
                'min': min_code,
                'max': max_code, 
                'list': np.zeros(max_code-min_code+1, dtype=int)
            }

            for i in range(max_code-min_code+1):  # min code should be negative
                self.histogram_dict['list'][i] = code_counter[i+min_code]

            print(self.histogram_dict)

            with open(self.histogram_path, 'wb') as f:
                pickle.dump(self.histogram_dict, f)

    def prob_to_cumulative_freq(self):
        laplace_smoothed = self._laplace_smooth(self.histogram_dict['list'])
        self.cum_freq = prob_to_cum_freq(laplace_smoothed, resolution=self.resolution)

    def get_min_code(self):
        return self.histogram_dict['min']

    def _laplace_smooth(self, histogram):
        # Returns the Laplace smoothed probability distribution
        for i in range(len(histogram)):
            histogram[i] += 100

        return histogram / sum(histogram)