
import csv
import re
import sys

DATA_BUNDLER = {"surprise" : "excited", "enthusiasm" : "excited", "fun":"excited",
                "hate" : "anger", "anger" : "anger", 
                "worry" : "worry",
                "sadness" : "sad", "empty" : "sad",
                "relief" : "relief",
                "love" : "love",
                "happiness" : "happy",
                "neutral" : "neutral"}

class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save):
        
        data = [] 
        emotions = {}

        with open(path_load, mode='r') as csv_data:

            csv_reader = csv.reader(csv_data, delimiter=',')
            line_count = 0
    
            for row in csv_reader:
    
                if line_count == 0:
                    line_count += 1
                    continue
    
                emotion = row[1]
                tweet = row[3]

                if ("\t" in tweet or " " in tweet): # Remove tabs and spaces and replace with one space
                    tweet = re.sub(r'[\t\s]+', ' ', tweet)
                
                # Strip @person from tweets
                if '@' in tweet:
                    # print(f"OLD: {tweet}")
                    tweet = re.sub(r'@[^\s]+(\s?)', "", tweet)
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


        with open(sys.argv[1]+".csv", "w+", newline='') as new_data:

            writer = csv.writer(new_data)
            writer.writerow(["emotion", "tweet"])

            for data_point in data:
                writer.writerow(data_point)


            print(f"Generated Dataset.")



if __name__ == '__main__':
    
    DataCleaner.generate_clean_dataset("../Data/text_emotion.csv", "../Data/text_emotion_cleaned.csv")