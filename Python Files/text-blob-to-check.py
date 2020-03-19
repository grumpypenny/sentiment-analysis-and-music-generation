from textblob import TextBlob
import sys
import csv

def interactive():
    end = False
    while not end:
        user_in = input("enter a sentence, or enter 'exit' to end:\n")
        if user_in == "exit":
            break
        sentiment = TextBlob(user_in)
        print(sentiment.sentiment[0])

def test():
    """
    Test the TextBlob AI using the s140 test dataset
    """
    correct = 0
    total = 0

    with open("../Data/s140_no_bias-test.csv", "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            total += 1
            label = row[0]
            tweet = row[1]
            
            s = TextBlob(tweet)
            s = s.sentiment[0]

            # print(s)
            # print(tweet)
            result = ""
            if (s < 0):
                result = "negative"
            else:
                result = "positive"

            if result == label:
                correct += 1

            if total % 10000 == 0:
                print("finished", total, "so far")

    return correct / total


if __name__ == "__main__":
    print("this prints the polarity of a given sentence")
    print("scale goes from -1.0 and 1.0")

    print("============================================")

    if len(sys.argv) == 0:
        print("USAGE: -i for interactive mode, -c to check against a csv file")
    else:
        mode = sys.argv[1]
        if mode == "-i":
            interactive()
        elif mode == "-c":
            print("test acc = ", test())
        else:
            print("USAGE: -i for interactive mode, -c to check against a csv file")


