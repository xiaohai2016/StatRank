"""
Unit testing codes
"""
import unittest
import data_loader


class TestLoadingMQ200xDataset(unittest.TestCase):
  """To test loading MQ2007/2008/MSLR-WEB10K/30K Listwise dataset"""

  def test_data_loader(self):
    """To test basic data loading code paths"""
    print('MQ2007:\n',
          data_loader.load_microsoft_dataset(
            'resources/MQ2007/Querylevelnorm.txt',
            max_entries=5))
    print('\nMQ2008:\n',
          data_loader.load_microsoft_dataset(
            'resources/MQ2008/Querylevelnorm.txt',
            max_entries=5))
    print('\nMSLR-WEB10K:\n',
          data_loader.load_microsoft_dataset(
            'resources/MSLR-WEB10K/Fold1/test.txt',
            feature_count=136,
            max_entries=5))
    print('\nMSLR-WEB30K:\n',
          data_loader.load_microsoft_dataset(
            'resources/MSLR-WEB30K/Fold5/test.txt',
            feature_count=136,
            max_entries=5))
    print('\nMQ2007 get_ms_dataset:\n',
          data_loader.get_ms_dataset(
            'resources/MQ2007/Querylevelnorm.txt',
            feature_count=46,
            scaler_id='MinMax',
            max_entries=5))
    print('\nMSLR-WEB30K get_ms_dataset:\n',
          data_loader.get_ms_dataset(
            'resources/MSLR-WEB30K/Fold5/test.txt',
            feature_count=136,
            scaler_id='Robust',
            max_entries=5))

if __name__ == '__main__':
  unittest.main()
