
import csv
import re
import sys
import random

S140 = [{'0' : "negative",
         '2' : "neutral",
         '4' : "positive"}, (0, 5, 4)]

CROWD = [{"surprise" : "happy", "enthusiasm" : "happy", "fun" : "happy", "happiness" : "happy", "love" : "happy",  # positive valence positive arousal
          "hate" : "anger", "anger" : "anger", 
          "worry" : "sad", "sadness" : "sad", "empty" : "sad", "boredom" : "sad",
          "relief" : "relief",
          "neutral" : "neutral"}, (1, 3)]

SHUFFLE_COUNT = 10 # Increase if paranoid about bad splits

BANNED_TEXT = ["sentiment", "neutral"]

def generate_clean_dataset(path_load, path_save, n, train_split, val_split, test_split):
    
    print("Starting data cleaning process...")

    data = [] 
    emotions = {}
    emotion_key, data_index = S140

    with open(path_load, mode='r') as csv_data:

        csv_reader = csv.reader(csv_data, delimiter=',')

        for row in csv_reader:
            
            emotion = row[data_index[0]]
            tweet = row[data_index[1]]
            author = None

            if len(data_index) > 2:
                author = row[data_index[2]]

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

            if emotion not in BANNED_TEXT:
                if emotion_key[emotion] not in emotions:
                    emotions[emotion_key[emotion]] = 0
                emotions[emotion_key[emotion]] += 1
                example = [emotion_key[emotion], tweet]
                if author:
                    example.append(author)
                data.append(example)

    print(f"Labels: {len(emotions)} Distribution: {emotions}")
    print(f"Processed {len(data)} examples in total.")

    print(f"Shuffling and splitting data...")

    for s in range(SHUFFLE_COUNT):
        random.shuffle(data)

    data_to_write = []
    if n > 0:
        data_to_write = data[0:n]
    else:
        data_to_write = data

    # Calculate split indices
    data_size = len(data_to_write)
    train_n = int(train_split * data_size)
    valid_n = int(val_split * data_size)
    test_n = int(test_split * data_size)

    # Slice dataset into splits
    train = data_to_write[:train_n]
    valid = data_to_write[train_n:(train_n+valid_n)]
    test = data_to_write[(train_n+valid_n):(train_n+valid_n+test_n)]

    if len(data_index) > 2: # Contains authors, more processing required
        
        print("Removing author bias...")

        train_authors = {}
        # Expected sizes
        actual_train_n = len(train)
        actual_valid_n = len(valid)
        actual_test_n = len(test)

        # Parition train data by author
        while len(train) > 0:
            example = train.pop()
            if example[2] not in train_authors:
                train_authors[example[2]] = []
            train_authors[example[2]].append(example)

        print(f"Found {len(train_authors)} author hits.")

        # Remove authors from validation dataset
        i = 0
        while i < len(valid):
            if valid[i][2] in train_authors: # Hit
                train_authors[valid[i][2]].append(valid.pop(i))
                continue
            i += 1

        # Remove authors from validation dataset
        i = 0
        while i < len(test):
            if test[i][2] in train_authors: # Hit
                train_authors[test[i][2]].append(test.pop(i))
                continue
            i += 1

        # Add back to train until we reach the split percentage
        keys = list(train_authors.keys())
        while len(train) < actual_train_n:
            train += train_authors.pop(keys.pop(), None)

        # Alternate between adding to test and validation
        # until we run out of examples
        a = False
        while len(keys) > 0:
            if a:
                test += train_authors.pop(keys.pop(), None)
            else:
                valid += train_authors.pop(keys.pop(), None)
            a = not a

        # Remove authors
        distribution = {"Train":{}, "Validation":{}, "Test":{}}
        for i in distribution:
            for emotion in list(emotion_key.values()):
                if emotion not in BANNED_TEXT:
                    distribution[i][emotion] = 0

        for i in range(len(train)):
            old = train[i]
            train[i] = [old[0], old[1]]
            distribution["Train"][old[0]] += 1
        for i in range(len(valid)):
            old = valid[i]
            valid[i] = [old[0], old[1]]
            distribution["Validation"][old[0]] += 1
        for i in range(len(test)):
            old = test[i]
            test[i] = [old[0], old[1]]
            distribution["Test"][old[0]] += 1

        print("Removed author bias.")
        print(f"Postprocess Distribution: {distribution}")

    # Save train set
    with open(path_save+"-train.csv", "w", encoding='utf-8', newline='') as new:
        wrtr = csv.writer(new)
        for point in train:
            wrtr.writerow(point)

        print(f"Wrote {len(train)} examples to {path_save}-train.csv.")

    # Save validation set
    with open(path_save+"-validation.csv", "w", encoding='utf-8', newline='') as new:
        wrtr = csv.writer(new)
        for point in valid:
            wrtr.writerow(point)

        print(f"Wrote {len(valid)} examples to {path_save}-validation.csv.")

    # Save test set
    with open(path_save+"-test.csv", "w", encoding='utf-8', newline='') as new:
        wrtr = csv.writer(new)
        for point in test:
            wrtr.writerow(point)

        print(f"Wrote {len(test)} examples to {path_save}-test.csv.")

    print("Done!")


if __name__ == '__main__':
    
    if len(sys.argv) != 7:
        print("Usage: [load_name] [save_name] [size] [train_split] [val_split] [test_split]")
        print("Let size = -1 to split entire dataset")
        exit(0)

    load_name = sys.argv[1]
    save_name = sys.argv[2]
    n = int(sys.argv[3])

    train, validation, test = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

    generate_clean_dataset(f"../Data/{load_name}.csv", f"../Data/{save_name}", \
                            n, train, validation, test)