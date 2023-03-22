
from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd


real_data = pd.read_csv("./heart_datagen.csv")


# Names of the columns that are discrete
discrete_columns = [
"output",
]


ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1197)
print(synthetic_data)

