if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import os


    import zipfile
    with zipfile.ZipFile("./sad/downsampled_1.zip","r") as zip_ref:
        zip_ref.extractall("./sad")
    pass