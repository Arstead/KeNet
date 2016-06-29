# -*- coding: utf-8 -*-
import numpy as np
from .utilor import Utilor
import gzip
import struct

class Mnist(Utilor):
    def __init__(self, *args, **kwargs):
        # some property of MNIST
        self.image_rows = 28
        self.image_cols = 28
        self.train_num = 60000
        self.test_num = 10000
        self.class_num = 10
        super(Mnist, self).__init__(name='mnist', *args, **kwargs)

    def load(self):
        TD_name = 'train-images-idx3-ubyte.gz'                                  # train data file name
        TD_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'  # train data url
        TL_name = 'train-labels-idx1-ubyte.gz'                                  # train labels file name
        TL_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'  # train labels url
        ED_name = 't10k-images-idx3-ubyte.gz'                                   # test data file name
        ED_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'   # test data url
        EL_name = 't10k-labels-idx1-ubyte.gz'                                   # test labels file name
        EL_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'   # test labels url

        # prepare data
        self._add_file(TD_name, TD_url)
        self._add_file(TL_name, TL_url)
        self._add_file(ED_name, ED_url)
        self._add_file(EL_name, EL_url)
        self._check_dir()
        self._download()

        # read train data
        TD_path = self._get_filepath(TD_name)
        file_in = gzip.GzipFile(TD_path)
        TD = self._read_data(file_in)
        file_in.close()
        del file_in

        # read train labels
        TL_path = self._get_filepath(TL_name)
        file_in = gzip.GzipFile(TL_path)
        TL = self._read_labels(file_in)
        file_in.close()
        del file_in

        # read test data
        ED_path = self._get_filepath(ED_name)
        file_in = gzip.GzipFile(ED_path)
        ED = self._read_data(file_in)
        del file_in

        # read test labels
        EL_path = self._get_filepath(EL_name)
        file_in = gzip.GzipFile(EL_path)
        EL = self._read_labels(file_in)
        del file_in

        return TD, TL, ED, EL

    def _read_labels(self, file_in):
        """
            read labels
        """
        struct.unpack('!i', file_in.read(4))  # remove magic number
        labels_num, = struct.unpack('!i', file_in.read(4))
        labels = struct.unpack('!%dB' % labels_num, file_in.read(labels_num))
        return np.array(labels)

    def _read_data(self, file_in):
        """
            read mnist image data
        """
        struct.unpack('!i', file_in.read(4))            # remove magic number
        sample_num, = struct.unpack('!i', file_in.read(4))
        rows_num, = struct.unpack('!i', file_in.read(4))
        cols_num, = struct.unpack('!i', file_in.read(4))
        pixels_num = rows_num * cols_num
        datas = np.zeros([sample_num, pixels_num])

        # I has been try to read all data in one time, which is about 47040000 bytes.
        # But python's default buffer size is 8192 bytes which is much smaller than that size.
        # However, I try to modify the default buffer size in python source code to 47040000.
        # And then it is working! That is amazing. But with a view to adapt most python users,
        # I use loop statement to read each image one by one.
        for i in range(sample_num):
            # read each image
            data = struct.unpack('!%dB' % pixels_num, file_in.read(pixels_num))
            data = np.array(data)
            datas[i] = data
        return datas


