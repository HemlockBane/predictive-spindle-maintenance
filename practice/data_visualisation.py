import matplotlib.pyplot as pyplot
import pandas
import seaborn

melbourne_file_path = "data/melb_data.csv"
data = pandas.read_csv(melbourne_file_path)

pyplot.figure(figsize=(16, 6))
seaborn.lineplot(data=data)
pyplot.show()
