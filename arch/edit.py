import csv

text = open("combined.csv", "r")
text = ''.join([i for i in text]) \
    .replace("https://web.archive.org/web/h", "https://web.archive.org/web/20091027*/h")
x = open("output.csv","w")
x.writelines(text)
x.close()