import pandas as pd
import numpy as np

weather = pd.read_csv("cleaned_data.csv", usecols=["Weather"])
unique_weather = weather['Weather'].unique()

# print(unique_weather)
# There are 33 unique types of weather. How common are each of them?

weather_counts = weather.groupby("Weather").size()
print(weather_counts.sort_values(ascending=False))

# "Cloudy" is the most common weather, followed by "Mostly Cloudy". Sounds like Vancouver! 

# Weather
# Cloudy                                          606
# Mostly Cloudy                                   572
# Rain                                            538
# Mainly Clear                                    532
# Clear                                           256
# Rain Showers                                    156
# Rain,Fog                                         91
# Snow                                             61
# Moderate Rain                                    20
# Drizzle                                          18
# Fog                                              18
# Moderate Rain,Fog                                11
# Snow Showers                                     10
# Drizzle,Fog                                       9
# Rain Showers,Fog                                  7
# Rain,Snow                                         7
# Rain,Drizzle,Fog                                  5
# Snow,Fog                                          5
# Moderate Snow                                     4
# Rain,Drizzle                                      4
# Moderate Rain Showers,Fog                         2
# Moderate Rain Showers                             2
# Moderate Rain,Drizzle                             2
# Heavy Rain,Fog                                    2
# Freezing Fog                                      2
# Thunderstorms                                     2
# Rain Showers,Snow Pellets                         1
# Heavy Rain,Moderate Hail,Fog                      1
# Heavy Rain Showers,Moderate Snow Pellets,Fog      1
# Heavy Rain                                        1
# Rain Showers,Snow Showers                         1
# Rain Showers,Snow Showers,Fog                     1
# Rain,Snow,Fog                                     1
