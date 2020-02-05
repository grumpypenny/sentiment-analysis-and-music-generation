
import csv


class DataCleaner:

    @staticmethod
    def generate_clean_dataset(path_load, path_save):
        
        data = [] 

        with open(path_load, mode='r') as csv_data:

            csv_reader = csv.reader(csv_data, delimiter=',')
            line_count = 0
            for row in csv_reader:
                emotion = row[1]
                tweet = row[3]

                # Strip @person from tweets
                


                data.append((emotion, tweet))
                line_count += 1





            print(f"Generated Dataset, Processed {line_count} data points.")
            print(data)


if __name__ == '__main__':
    
    DataCleaner.generate_clean_dataset("../Data/text_emotion.csv", "../Data/text_emotion_cleaned.csv")