import os
from ..core import config
import urllib.request


class Utilor(object):
    def __init__(self, name, base_path=config.data_base_path, *args, **kwargs):
        self.name = name
        self.base_path = base_path
        self.dir_name = name
        self.file_info_list = []
        self.train_num = 0
        self.test_num = 0
        self.class_num = 0
        return super(Utilor, self).__init__(*args, **kwargs)

    def _check_dir(self):
        dir_path = self.get_dirpath()
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def _check_file(self, file_path):
        return os.path.exists(file_path)

    def _get_filepath(self, file_name):
        return os.path.join(self.base_path, self.dir_name, file_name)

    def get_filespath(self):
        paths = []
        for (file_name, file_url) in self.file_info_list:
            paths.append(self._getfilepath(file_name))
        return paths

    def get_dirpath(self):
        return os.path.join(self.base_path, self.dir_name)

    def _download(self):
        for (file_name, file_url) in self.file_info_list:
            file_path = self._get_filepath(file_name)
            if not self._check_file(file_path):
                print('File \'%s\' isn\'t exist, please wait few time for us to download...' % file_name)
                assert file_url is not None, 'File %s isn\'t exist, which requests an url' % file_name
                urllib.request.urlretrieve(file_url, file_path)
                print('File \'%s\' downloads finish')

    def _add_file(self, file_name, file_url=None):
        self.file_info_list.append((file_name, file_url))

    def load(self):
        pass

    def _read_label(self, file_path):
        pass

    def _read_data(self, file_path):
        pass


if __name__ == '__main__':
    utilor = Utilor('mnist')
    utilor.load()

