import glob
import os
import re
import csv

def get_data():
    # for file in glob.glob("./ABC_cleaned/*.abc"):
    all_files = []

    start = re.compile("^X:(\s)?\d+")

    path = "D:/School/_ML_project/abc/Tune-a-day"

    encoding = 'utf-8'

    for file in os.listdir( os.fsencode(path)):
        fname = os.fsdecode(file)
        print(fname)
        if fname.endswith(".abc"):
            tunes = [] # will be a list of lists

            with open(path+'/'+fname, "r", encoding=encoding, newline="", errors='ignore') as f:
                # store all the tunes into a list
                should_append = False
                tune = []
                for line in f.readlines():
                    # remove any non utf-8 characters
                    line = bytes(line, encoding).decode(encoding, 'ignore')
                    # line is 'X: dd'
                    if start.match(line):
                        should_append = True
                        # finish with the old group
                        # start a new one
                        tunes.append(tune)
                        tune = []
                    if should_append:
                        tune.append(line)
                # add the last one
                tunes.append(tune)
            
            # remove any empty groups (ie if line 1 was empty)
            for group in tunes:
                if not group:
                    tunes.remove(group)   
            all_files.append(tunes)

    return all_files

def confirm():
    real = []
    read = []
    with open("./ABC_cleaned/ashover.abc", "r", encoding="utf-8", newline="") as f:
        real = f.readlines()
    with open("confirm.txt", "r", encoding="utf-8", newline="") as c:
        read = c.readlines()
    for i in range(len(real)):
        if not (real[i] == read[i]):
            print("real: ", real[i], end="")
            print("READ: ", read[i], end="")
            exit(0)
    return real == read

def get_list(tune_list):
    out = []
    for tunes in tune_list:
        for tune in tunes:
            string = ''.join(tune)
            # print("====================================================")
            # print(string, end="")
            # print("====================================================")
            out.append(string)
    return out

def make_csv(string_list):
    """ 
    Take in a list of strings reprsenting a tune
    Then put it in a csv
    """
    with open("tunes.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        for i, s in enumerate(string_list):
            writer.writerow([s])


if __name__ == "__main__":
    tune_list = get_data()

    data = get_list(tune_list)

    make_csv(data)
      

