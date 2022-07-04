# Preprocess the ERA5 dataaset and convert to .npy files
# The user will define the following (as arg):
#   1. Pressure level (-1 for surface)
#   2. GRIB file DIR
#   3. Variable to extract
# The output will be the .npy files in the predefined DIR for the dataset.
#       We only extract the variable at lead-time T=48
