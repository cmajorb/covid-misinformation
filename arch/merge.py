import os

base = "results/"
count = 0
with open('combined.csv','w') as mainfile:
    mainfile.write('ID,URL,aids,hiv,grids,aids/hiv,virus\n')

    for f in os.listdir(base):
        with open(base + f, "r") as infile:
            for line in infile:
                mainfile.write(line)
