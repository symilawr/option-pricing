import pandas as pd

class Analyze:

    def __init__(self) -> None:
        pass

    def calc_std_dev(self, data):
        data = pd.DataFrame(data=data)
        standard_deviation = data.iloc[:, 2].std()
        return standard_deviation
    
    