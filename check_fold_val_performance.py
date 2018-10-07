import pandas as pd
import sys, os


traininglogs = sys.argv[1:]
model_nums = [t.split("/")[-1] for t in traininglogs]
traininglogs = [pd.read_csv(traininglog) for traininglog in traininglogs]

fmt = " "*44 + (5*" ").join(map(lambda x: "{:>14s}".format(x), model_nums))
print(fmt,"_"*len(fmt),sep="\n")
for col in filter(lambda col: "error" in col, traininglogs[0].columns):
    mins = [min(traininglog[col].values.tolist()) for traininglog in traininglogs]
    print("{:<44s}".format(col) + (5*" ").join(list(map(lambda x: "{0:>14.3f}".format(round(x,3)), mins))))
