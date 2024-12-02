"""
Script for opening and testing CIFAR dataset
"""

CIFAR_FILEPATH = './data/cifar-10'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class DataLoader(object):
    """
    Loader for various sources of data
    """
    def __init__(self, dataset: str) -> None:
        self.num_batches = None
        self.data = None
        self.labels = None

        # Add support for different datasets here
        if dataset.lower() == 'cifar':
            self.load_cifar()
        else:
            raise NotImplementedError("Dataset not supported")

    
    def load_cifar(self, batch_index=None):
        """
        batches.meta structure:
        {
            b'num_cases_per_batch': 10000,
            b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'],
            b'num_vis': 3072
        }
        """
        self.num_batches = 5

        self.data = [None] * self.num_batches
        self.labels = [None] * self.num_batches

        metadata = unpickle(f'./{CIFAR_FILEPATH}/batches.meta')
        print(f'Metadata')
        print(metadata)

        """
        sample_batch structure:
        {
            data: 10000 x 3072 numpy array 
                (32 x 32 RGB image, [0:1024] R, [1024:2048] G, [2048:3072] B). Stored in row major order
            labels = 1 x 10000 list. 
                labels[i] = 0-9
                Label indices correspond to label classes found in 'batches.meta' 
        }
        """

        # Return data/labels for specific batch
        if batch_index is not None:
            batch_data_dict = unpickle(f'{CIFAR_FILEPATH}/data_batch_{batch_index}')
            self.data[0] = batch_data_dict[b'data']
            self.labels[0] = batch_data_dict[b'labels']
        else:
            for batch_i in range(self.num_batches):
                batch_data_dict = unpickle(f'{CIFAR_FILEPATH}/data_batch_{batch_i + 1}')
                self.data[batch_i] = batch_data_dict[b'data']
                self.labels[batch_i] = batch_data_dict[b'labels']


        
        