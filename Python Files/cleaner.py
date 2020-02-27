
import csv
import re
import sys
import random

S140 = [{'0' : "negative",
         '2' : "neutral",
         '4' : "positive"}, (0, 5)]

CROWD = [{"surprise" : "happy", "enthusiasm" : "happy", "fun" : "happy", "happiness" : "happy", "love" : "happy",  # positive valence positive arousal
          "hate" : "anger", "anger" : "anger", 
          "worry" : "sad", "sadness" : "sad", "empty" : "sad", "boredom" : "sad",
          "relief" : "relief",
          "neutral" : "neutral"}, (1, 3)]

SHUFFLE_COUNT = 10

class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save, n, train_split, val_split, test_split):
        
        if (train_split + val_split + test_split) != 1:
            print("The split does not sum to 1.")
            return

        data = [] 
        emotions = {}
        key, data_index = CROWD

        with open(path_load, mode='r') as csv_data:

            csv_reader = csv.reader(csv_data, delimiter=',')
            line_count = 0
    
            for row in csv_reader:
                
                emotion = row[data_index[0]]
                tweet = row[data_index[1]]

                # make all lower case?
                tweet = tweet.lower()

                # Remove tabs and spaces and replace with one space
                if ("\t" in tweet or " " in tweet): 
                    tweet = re.sub(r'[\t\s]+', ' ', tweet)
                
                # remove any links
                tweet = re.sub(r'\bhttp(s?):\/\/[^\s]+', "", tweet)
                
                # remove all non-letters/numbers and spaces
                tweet = re.sub(r'[^0-9a-z,\s]+', "", tweet)

                # Strip @person from tweets
                if '@' in tweet:
                    # remove @handle
                    # remove @ handle
                    tweet = re.sub(r'(@\s[^\s]+\s)|(@[^\s]+(\s?))', "", tweet)

                    if not tweet:
                        continue

                if emotion != "sentiment" and emotion != "neutral":
                    if key[emotion] not in emotions:
                        emotions[key[emotion]] = 0
                    emotions[key[emotion]] += 1
                    data.append([key[emotion], tweet])

                line_count += 1

            print(emotions, len(emotions))
            print(f"Processed {len(data)} data points.")

            for s in range(SHUFFLE_COUNT):
                random.shuffle(data)

            data_to_write = data[0:n]

            k = len(data_to_write)
            train_n = int(train_split * k)
            valid_n = int(val_split * k)
            test_n = int(test_split * k)

            train = data_to_write[:train_n]
            valid = data_to_write[train_n:(train_n+valid_n)]
            test = data_to_write[(train_n+valid_n):(train_n+valid_n+test_n)]

        # Save train set
        with open(path_save+"-train.csv", "w", encoding='utf-8', newline='') as new:
            wrtr = csv.writer(new)
            for point in train:
                wrtr.writerow(point)

            print(f"Wrote {len(train)} data points to {path_save}-train.csv.")

        # Save validation set
        with open(path_save+"-validation.csv", "w", encoding='utf-8', newline='') as new:
            wrtr = csv.writer(new)
            for point in valid:
                wrtr.writerow(point)

            print(f"Wrote {len(valid)} data points to {path_save}-validation.csv.")

        # Save test set
        with open(path_save+"-test.csv", "w", encoding='utf-8', newline='') as new:
            wrtr = csv.writer(new)
            for point in test:
                wrtr.writerow(point)

            print(f"Wrote {len(test)} data points to {path_save}-test.csv.")


if __name__ == '__main__':
    
    load_name = sys.argv[1]
    save_name = sys.argv[2]
    n = int(sys.argv[3])

    train, validation, test = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

    DataCleaner.generate_clean_dataset(f"../Data/{load_name}.csv", f"../Data/{save_name}", \
                                        n, train, validation, test)