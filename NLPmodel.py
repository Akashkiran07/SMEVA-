from rake_nltk import Rake
import nltk

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

rake = Rake()

def nlp_check(ans):
    
    file_text = open("text.txt","r")
    text = file_text.read()

    rake.extract_keywords_from_text(text)
    kws = rake.get_ranked_phrases()

    # ans_file = open("ans.txt","r")
    # ans = ans_file.read()

    for i in kws:
        if (" " in i):
            kws.remove(i)
            i = i.split(' ')
            for x in i:
                kws.append(x)

    print(kws)

    marks = 0

    for i in kws:
        if(i in ans):
            marks += 1
            print(i)

    score = (marks/len(kws))
    
    if(marks >= 5):
        score = 1
        
    print('Score', score)
    print('Marks', marks )
    
    return score
