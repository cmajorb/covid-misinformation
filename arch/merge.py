import os

base = "results2/"
count = 0
with open('combined2.csv','w') as mainfile:
    #mainfile.write('ID,URL,aids,hiv,grids,aids/hiv,virus\n')
    #mainfile.write('ID,URL,text\n')
    for f in os.listdir(base):
        with open(base + f, "r") as infile:
            header = False
            for line in infile:
                if header:
                    mainfile.write(line)
                header = True
