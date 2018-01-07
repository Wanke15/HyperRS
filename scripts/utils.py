def transform(image, filter_size=5, to_int=False):
    import numpy as np
    from scipy.misc import imread
    
    if filter_size%2==0:
        print('Filter_size has be odd!!!')
        return
        
    #image = imread(image)

    row_num = int(image.shape[0]+(filter_size-1))
    col_num = int(image.shape[1]+(filter_size-1))
    
    image_pad = np.zeros((row_num, col_num, int(image.shape[2])), dtype = np.float32)
    
    row_start = int((filter_size-1)/2)
    row_end = int(image.shape[0]+(filter_size-1)/2)
    
    col_start = int((filter_size-1)/2)
    col_end = int(image.shape[1]+(filter_size-1)/2)
    
    #print(row_start)
    
    image_pad[row_start:row_end, col_start:col_end] = image
    
    dataset = np.zeros((image.shape[0]*image.shape[1], filter_size, filter_size, image.shape[2]), dtype = np.float32)
    print("dataset shape:",dataset.shape)
    print('loop number:',(image_pad.shape[0]-(filter_size-1))*(image_pad.shape[1]-(filter_size-1)))
    
    for i in range(image_pad.shape[0]-(filter_size-1)):
        for j in range(image_pad.shape[1]-(filter_size-1)):
            dataset[i*image.shape[1]+j] = image_pad[i:i+filter_size, j:j+filter_size]
    if to_int:
        dataset = dataset.astype('uint8')
        return dataset
    else:
        return dataset


def load_data(data_path):
    import numpy as np
    dataset = np.load(load_data)

    
