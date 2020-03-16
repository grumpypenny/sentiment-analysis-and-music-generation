import os
import sys
import csv
import string

happy = []
sad = []


for file in os.listdir( os.fsencode(".") ):
    fname = os.fsdecode(file)
    if fname.endswith("csv"):
        # print(fname)
        with open (fname, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)

            for line in reader:
                lst = []
                # print(line[0])
                string = line[0]
                key = string.find("K:")
                key = key + 2 # line[0][key] = 'K' we want the thing after the colon
                for char in string[key:]:
                    if char != '\n' and char != '%' and char != '\r':
                        lst.append(char)
                    else:
                        break
                out = "".join(lst)
                out = out.strip(" ")
                """
                https://www.youtube.com/watch?v=jNY_ZCUBmcA
                use that video to understand modes
                Darkest -> Brightest
                Locrian, Phrygian, Minor, Dorian, Mixolydian, Major, Lydian
                                                              ^ here is the happy boundry
                Most of the data is major
                """
                if (len(out) == 1 or 
                        "maj" in out or 
                        "Maj" in out or
                        ("lyd" in out.lower() and "mixolydian" not in out.lower())):
                    # major by default or has maj/Maj in it
                    happy.append(line)
                else:
                    sad.append(line)
                    print("|", out, "|", end="")
                    print(" ", len(out))
                # print(out, end="")
                # print(" ", len(out))

with open("happy.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    for s in happy:
        writer.writerow(s)

with open("sad.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    for s in sad:
        writer.writerow(s)
    


