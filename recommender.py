import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Suppress numpy bug
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Returns all books the user has rated
def getRatedBooks(userId, ratings):
    return ratings.query('`User-ID` == @userId')[['ISBN', 'Book-Rating']]

# Returns all ISBNs of the books the user has rated
def getRatedISBN(rated):
    ISBN = []
    for row in rated.itertuples(name=None):
        ISBN.append(row[1]) 
    return ISBN

# Returns the top 3 books scores
# Format: (ISBN, score)
def getTopThreeRatings(rated):
    bookRatings = []
    
    userRatings = rated.sort_values(by=['Book-Rating'], ascending=False)
    userRatings = userRatings.head(3)

    for row in userRatings.itertuples(name=None):
        bookRatings.append( (row[1], row[2]) )
        
    return bookRatings

# Returns all details for the top 3 books given the ratings
# Format: (Title, Author, Year of publication)
def getTopThreeBooks(topThreeRatings, books):
    topThreeBooks = []
    for isbn, rating in topThreeRatings:
        bookData = books.query('ISBN == @isbn')[['Book-Title', 'Book-Author', 'Year-Of-Publication']]
        for row in bookData.itertuples(name=None):
            topThreeBooks.append( (row[1], row[2], row[3]) )
    return topThreeBooks

# Returns all keywords for a given book title
def getKeywords(title):
    keywords = []
    
    stop_words = set(stopwords.words('english'))
    blacklist = ["paperback", "novel", "vol", "hardcover", "edition"]
    
    title = title.lower()

    # Remove all symbols
    title = re.sub(r'[^\w]', ' ', title)
    
    tokens = word_tokenize(title)
    '''
    Append a token to keywords if:
    1) It's not a stop word.
    2) It's not in the blacklist
    3) It's not already in keywords
    4) It has at least 2 characters
    '''
    for t in tokens:
        if (t not in stop_words and t not in blacklist and
            t not in keywords and len(t) > 1):
            keywords.append(t)
    return keywords

# Returns a list of all keywords combined from the top 3 book titles
def getTopThreeKeywords(topThreeBooks):
    keywords = [] 
    for book in topThreeBooks:
        keywords.extend(getKeywords(book[0]))
    return keywords

# Returns a list of all authors from the top 3 books
def getTopThreeAuthors(topThreeBooks):
    authors = []
    for book in topThreeBooks:
        authors.append(book[1].lower())
    return authors

# Returns a list of publication years from the top 3 books
def getTopThreeYears(topThreeBooks):
    years = [] 
    for book in topThreeBooks:
        years.append(book[2])
    return years

# Calculates Jaccard similarity
# Note: It's possible to get a ZeroDivisionError due to missing data info
def calcJaccard(a, b):
    s1 = set(a)
    s2 = set(b)
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

# Calculates Dice coefficient
def calcDice(a, b):
    s1 = set(a)
    s2 = set(b)
    overlap = len(s1 & s2)
    return overlap * 2.0/(len(s1) + len(s2))

# Returns the dice coefficient and jaccard score for a book
def similarity(userKeywords, userAuthors, userYears, title, author, year, keywords):    
    jaccardScore = 0.0
    diceScore = 0.0
    
    jaccardScore += 0.2 * calcJaccard(keywords, userKeywords)
    diceScore += 0.5 * calcDice(keywords, userKeywords)
    if author in userAuthors:
        jaccardScore += 0.4
        diceScore += 0.3

    bestYearMultiplier = 0.0
    for userYear in userYears:
        multiplier = 1 - (abs(int(userYear) - year) / 2005)
        if multiplier > bestYearMultiplier:
            bestYearMultiplier = multiplier

    jaccardScore += 0.4 * bestYearMultiplier
    diceScore += 0.2 * bestYearMultiplier
    
    return jaccardScore, diceScore

# Returns the golden standard list for Question 3
# Format: (ISBN, Title, Author, Year, Occurences, Total score)
def getGoldenStandard(topJaccard, topDice):
    golden = []
    topGolden = []
    checkedISBN = []

    # Check if a book in the jaccard list exists in the dice list
    for book in topJaccard:
        ISBN = book[0]
        match = [item for item in topDice if item[0] == ISBN]
        # If there's a book, calculate the new score and append it with occurences = 2
        if match:
            totalScore = (book[4] + match[0][4]) / 2
            golden.append( (book[0], book[1], book[2], book[3], 2, totalScore) )
        else:
            # No match, occurences = 1
            golden.append( (book[0], book[1], book[2], book[3], 1, book[4]) ) 
        checkedISBN.append(ISBN)

    # Add all books from the Dice list with only 1 occurence
    for book in topDice:
        ISBN = book[0]
        if ISBN in checkedISBN:
            continue
        golden.append( (book[0], book[1], book[2], book[3], 1, book[4]) )

    # Sort the golden list, first with occurences, then with scores
    # Negate the score to imply reverse=True
    golden.sort(key = lambda x: (x[4], -x[5]))

    # Append top 10 to the final list and return it
    for i in range(10):
        topGolden.append(golden[i])
        
    return topGolden
    
print("Loading...")

# We need all the books to get the info of the 3 higher rated books per user
books = pd.read_csv("BX-Books.csv", encoding = "ansi", sep = ";", low_memory = False, index_col=0)
# Modified data
filteredBooks = pd.read_csv("books.csv", encoding = "ansi", sep = ";", low_memory = False, index_col=0)
users = pd.read_csv("users.csv", encoding = "ansi", sep = ";", index_col=0)
ratings = pd.read_csv("ratings.csv", encoding = "ansi", sep = ";", index_col=0)

print("Loading done")
print("Recommending for 5 users...")

# Pick 5 random users
randomUsers = users.sample(5)

for i, row in enumerate(randomUsers.itertuples(name=None)):
    userId = row[0]

    # Get all books the user has rated
    ratedBooks = getRatedBooks(userId, ratings)

    # Get all ISBNs of the rated books
    ISBN = getRatedISBN(ratedBooks)

    # Get top 3 books
    topThreeRatings = getTopThreeRatings(ratedBooks)
    topThreeBooks = getTopThreeBooks(topThreeRatings, books)
    print('User ' + str(i+1), topThreeBooks)
    print()

    # Get the keywords, authors, publication years from the top 3 books
    userKeywords = getTopThreeKeywords(topThreeBooks)
    userAuthors = getTopThreeAuthors(topThreeBooks)
    userYears = getTopThreeYears(topThreeBooks)

    gradedBooksJaccard = []
    gradedBooksDice = []

    # Scan all filtered books
    for book in filteredBooks.itertuples(name=None):
        if book[0] in ISBN:
            continue

        ISBN, title, author, year =  book[0], book[1], book[2], book[3]
        keywords = getKeywords(title)

        # Calculate dice, jaccard score for every book
        jaccard, dice = similarity(userKeywords, userAuthors, userYears, title, author.lower(), year, keywords)
        gradedBooksJaccard.append( (ISBN, title, author, year, jaccard) )
        gradedBooksDice.append( (ISBN, title, author, year, dice) )

    # Sort both lists   
    gradedBooksJaccard.sort(key = lambda x: x[4], reverse = True)
    gradedBooksDice.sort(key = lambda x: x[4], reverse = True)

    topJaccard = []
    topDice = []
    averageOverlap = 0.0
    # Pick the top 10 scored books of every list and
    # calculate the average for question 2
    for j in range(10):
        topJaccard.append(gradedBooksJaccard[j])
        topDice.append(gradedBooksDice[j])
        if (gradedBooksJaccard[j][0] == gradedBooksDice[j][0]):
            averageOverlap += 1.0
    averageOverlap /= 10

    # Write question 1 to file
    df = pd.DataFrame(topJaccard, columns=['ISBN', 'Title', 'Author', 'Year', 'Jaccard'])
    df.to_csv('JaccardResult' + str(i+1) + '.csv', index=False)

    df = pd.DataFrame(topDice, columns=['ISBN', 'Title', 'Author', 'Year', 'Dice'])
    df.to_csv('DiceResult' + str(i+1) + '.csv', index=False)

    # Write question 2 to file
    with open('ListSimilarity_' + str(i+1) + '.txt', 'w') as file:
        file.write('List similarity: ' + str(averageOverlap))
    
    # Question 3
    goldenList = getGoldenStandard(topJaccard, topDice)
    
    df = pd.DataFrame(goldenList, columns=['ISBN', 'Title', 'Author', 'Year', 'Occurences', 'Total Score'])
    df.to_csv('GoldenResult' + str(i+1) + '.csv', index=False)

    avgGoldenJaccard = 0.0
    avgGoldenDice = 0.0

    # Averages for the golden list 
    for j in range(10):
        if (goldenList[j][0] == topJaccard[j][0]):
            avgGoldenJaccard += 1.0
        if (goldenList[j][0] == topDice[j][0]):
            avgGoldenDice += 1.0
        
    avgGoldenJaccard /= 10
    avgGoldenDice /= 10

    with open('GoldenSimilarity_' + str(i+1) + '.txt', 'w') as file:
        file.write('Golden-Jaccard similarity: ' + str(avgGoldenJaccard) + '\n')
        file.write('Golden-Dice similarity: ' + str(avgGoldenDice))
        
print("Recommendation complete")
