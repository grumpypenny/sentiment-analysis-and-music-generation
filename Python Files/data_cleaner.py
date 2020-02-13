
import csv
import re

class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save):
        
        data = [] 
        tweet_acc = {}

        with open(path_load, mode='r') as csv_data:

            csv_reader = csv.reader(csv_data, delimiter=',')
            line_count = 0
            for row in csv_reader:
                emotion = row[1]
                tweet = row[3]

                if ("\t" in tweet or " " in tweet): # Remove tabs and spaces and replace with one space
                    tweet = re.sub(r'[\t\s]+', ' ', tweet)

                if (row[2] in tweet_acc):
                    print(f"DUPLICATE USER {row[2]}")
                    print(f"TWEET 1: {tweet_acc[row[2]]}")
                    print(f"TWEET 2: {tweet}\n")

                tweet_acc[row[2]] = tweet

                # Strip @person from tweets
                


                data.append((emotion, tweet))
                line_count += 1





            print(f"Generated Dataset, Processed {line_count} data points.")
            # print(data)


if __name__ == '__main__':
    
    DataCleaner.generate_clean_dataset("../Data/text_emotion.csv", "../Data/text_emotion_cleaned.csv")