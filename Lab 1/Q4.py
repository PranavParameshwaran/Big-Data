import statistics

sample = [67.65, 167, 172, 175, 165, 167, 168, 167, 167.3, 170, 167.5, 170, 167, 169, 172]

mean = statistics.mean(sample)
median = statistics.median(sample)
mode = statistics.mode(sample)
stddev = statistics.stdev(sample)
skewness = (mean - mode)/stddev

print(sample)
print("mean = ",mean,"\nmedian = ",median,"\nmode = ",mode,"\nstddev = ",stddev,"\nskewness = ",skewness)