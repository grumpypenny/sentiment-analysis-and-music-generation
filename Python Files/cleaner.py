
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

class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save, n):
        
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

            random.shuffle(data)

        with open(path_save, "w", encoding='utf-8', newline='') as new:
            wrtr = csv.writer(new)
            i = 0
            for point in data:
                wrtr.writerow(point)
                i += 1
                if i == n:
                    break

            print(f"Wrote {i} data points.")


if __name__ == '__main__':
    
    load_name = sys.argv[1]
    save_name = sys.argv[2]
    n = int(sys.argv[3])
    DataCleaner.generate_clean_dataset(f"../Data/{load_name}.csv",
     f"../Data/{save_name}.csv", n)