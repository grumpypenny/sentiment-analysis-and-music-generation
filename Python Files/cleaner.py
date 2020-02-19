
import csv
import re
import sys

DATA_BUNDLER = {'0' : "negative",
                '2' : "neutral",
                '4' : "positive"}

# PATH_TO_DATA: "D:\School\_ML_project\sentiment\sentiment140"

class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save):
        
        data = [] 
        emotions = {}

        with open(path_load, mode='r') as csv_data:

            csv_reader = csv.reader(csv_data, delimiter=',')
            line_count = 0
    
            for row in csv_reader:
        
                emotion = row[0]
                tweet = row[5]
                
                # Remove tabs and spaces and replace with one space
                if ("\t" in tweet or " " in tweet): 
                    tweet = re.sub(r'[\t\s]+', ' ', tweet)
                
                # remove any links
                tweet = re.sub(r'\bhttp(s?):\/\/[^\s]+', "", tweet)
                
                # Strip @person from tweets
                if '@' in tweet:
                    # remove @handle
                    # remove @ handle
                    tweet = re.sub(r'(@\s[^\s]+\s)|(@[^\s]+(\s?))', "", tweet)

                    if not tweet:
                        continue
                    # print(f"NEW: {tweet}\n")

                ## Make All Lowercase? ##

                if emotion != "sentiment" and emotion != "boredom":
                    if DATA_BUNDLER[emotion] not in emotions:
                        emotions[DATA_BUNDLER[emotion]] = 0
                    emotions[DATA_BUNDLER[emotion]] += 1
                    data.append([DATA_BUNDLER[emotion], tweet])

                line_count += 1

            print(emotions, len(emotions))
            print(f"Processed {line_count} data points.")


        with open(path_save, "w", encoding='utf-8', newline='') as new:
            wrtr = csv.writer(new)

            for point in data:
                wrtr.writerow(point)

            print("done")



if __name__ == '__main__':
    
    DataCleaner.generate_clean_dataset("../Data/testdata.manual.2009.06.14.csv",
     "../Data/s140_500tweets.csv")