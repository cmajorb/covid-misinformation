import os

base = "results/"
count = 0
with open('combined.csv','w') as mainfile:
    mainfile.write('ID,tweet_id,input,output\n')
    for f in os.listdir(base):
        with open(base + f, "r") as infile:
            header = False
            for line in infile:
                new_line = line.strip()
                if header:
                    mainfile.write(new_line + '\n')
                header = True
                count += 1
                if count%100 == 0:
                    print(new_line)