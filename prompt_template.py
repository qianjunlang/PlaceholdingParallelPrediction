
def get_ans( dataset_name , sample ):

    if dataset_name == "amazon_review_polarity":
        cls_name = ["negative","positive"]
        return cls_name[sample[1]-1]
    
    if dataset_name == "aclImdb":
        return sample[2]
    
    if dataset_name == "dbpedia":
        cls_name = ["Company","EducationalInstitution","Artist","Athlete","OfficeHolder","MeanOfTransportation","Building","NaturalPlace","Village","Animal","Plant","Album","Film","WrittenWork",]
        return cls_name[sample[1]-1]
    
    if dataset_name == "ag_news":
        cls_name = ["world","sports","business","technology"]
        return cls_name[sample[1]-1]
    
    if dataset_name == "yahoo_answers":
        cls_name = ["Society & Culture","Science & Mathematics","Health","Education & Reference","Computers & Internet","Sports","Business & Finance","Entertainment & Music","Family & Relationships","Politics & Government",]
        return cls_name[sample[1]-1]
    
    if dataset_name == "SST2":
        cls_name = ["negative","positive"]
        return cls_name[sample[1]]
    
    if dataset_name == "isear":
        return sample[1]
    
    if dataset_name == "carer":
        return sample[1]
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

prompt_templates = {

    "amazon_review_polarity": lambda title, text: {
        "dataset_prompt" : [ 
            f'{title}\n{text}\nMy feedback to it is ',
            f'{title}\n{text}\nOverall, my feedback to it is ',
            f'{title}\n{text}\nAfter considering all aspects, my feedback to it is ',
            f'{title}\n{text}\nAfter considering all aspects, my viewpoint is ',
            f'{title}\n{text}\nReflecting on the above, my viewpoint is ',
            f'{title}\n{text}\nOverall, my perspective on it is ',
            f'{title}\n{text}\nOverall, my takeaway is ',
            f'{title}\n{text}\nIn summary, I would say ',

            f'{title}\n{text}\nConsidering the details provided, my emotional reaction is ',
            f'{title}\n{text}\nTaking into account the experience shared, my viewpoint is ',
            f'{title}\n{text}\nReflecting on the content, my emotional stance is ',
            f'{title}\n{text}\nGiven the information above, my perspective on it is ',
            f'{title}\n{text}\nAnalyzing the feedback, my emotional assessment is ',
            f'{title}\n{text}\nBased on the review, my overall sentiment impression is ',
            f'{title}\n{text}\nWeighing up the insights, my sentiment conclusion is ',
            f'{title}\n{text}\nAfter thoroughly considering the review, my sentiment perspective is ',
        ],
        "question_prompt" : [
            f"Q: Here is an Amazon review: {text} \nHelp me to determine whether it is positive or negative?\nA:",
            f"Q: Given an Amazon review: {text} \nIs the sentiment of this review positive or negative?\nA:",
            f"Q: Please analyze the following review found on Amazon: {text} \nBased on the content and tone of the comment, do you think the reviewer's attitude towards the product is positive or negative?\nA:",

        ],
        "ICL_prompt" : [
            f'Text: {title} {text}\nSentiment:',
            f'Text: {title} {text}\nSentiment Analysis: The overall sentiment is ',
        ],
        "sum_prompt" : [
            f'{title}\n{text}\nAll in all, it was ',
            f'{title}\n{text}\nIn summary, it was ',
            f'{title}\n{text}\nIn essence, it was ',
            f'{title}\n{text}\nIn conclusion, it was ',
            f'{title}\n{text}\nTo sum up, it\'s ',
        ],
        "short_prompt" : [
            f'{title}\n{text} All in all ',
            f'{title}\n{text} Just ',
            f'{title}\n{text} It was ',
            f'{title}\n{text} It is ',
            f'{title}\n{text} That is ',
            f'{title}\n{text} That\'s ',
            f'{title}\n{text} But it is ',
        ],
        "empty_prompt" : [
            f'{title}\n{text} '
        ],
    },
    
    "aclImdb": lambda text :  {
        "dataset_prompt" : [
            f'{text}\nMy feedback to the film is ',
            f'{text}\nOverall, my feedback to the film is ',
            f'{text}\nAfter considering all aspects, my feedback to the film is ',
            f'{text}\nAfter considering all aspects, my viewpoint is ',
            f'{text}\nReflecting on the above, my viewpoint is ',
            f'{text}\nOverall, my perspective on the film is ',
            f'{text}\nOverall, my takeaway is ',
            f'{text}\nIn summary, I would say ',
            f'{text}\nI think it is ',
            f'{text}\nOverall, I think it is ',
            f'{text}\nConsidering everything, my feedback is',
            f'{text}\nConsidering everything, I think it is ',
            f'{text}\nAfter thinking about it, my feedback is',
            f'{text}\nOverall, I see it as ',
            f'{text}\nTaking all factors into account, my assessment of it is',
            
            f'{text}\nConsidering the details provided, my emotional reaction is ',
            f'{text}\nTaking into account the experience shared, my viewpoint is ',
            f'{text}\nReflecting on the content, my emotional stance is ',
            f'{text}\nGiven the information above, my sentiment evaluation is ',
            f'{text}\nAnalyzing the feedback, my emotional assessment is ',
            f'{text}\nBased on the review, my overall sentiment impression is ',
            f'{text}\nWeighing up the insights, my sentiment conclusion is ',
            f'{text}\nAfter thoroughly considering the review, my sentiment perspective is ',
        ],
        "question_prompt" : [
            
            f"Q: Here is a film review:{text} \nHelp me to detemine whether it is positive or negative?\nA:",

            f"Q: Given an IMDb review: {text} \nIs the sentiment of this review positive or negative?\nA:",
            f"Q: Please analyze the following review found on IMDb: {text} \nBased on the content and tone of the comment, do you think the reviewer's attitude towards the movie positive or negative?\nA:", #!new
            
        ],
        "ICL_prompt" : [
            f'Text: {text}\nSentiment:',
            f'Text: {text}\nSentiment Analysis: The overall sentiment is ',
        ],
        "sum_prompt" : [
            f'{text}\nAll in all, the film was ',
            f'{text}\nIn summary, the film was ',
            f'{text}\nIn essence, the film was ',
            f'{text}\nIn conclusion, the film was ',
            f'{text}\nTo sum up, the film was ',
        ],
        "short_prompt" : [
            f'{text} All in all ',
            f'{text} Just ',
            f'{text} It was ',
            f'{text} It is ',
            f'{text} That is ',
            f'{text} That\'s ',
            f'{text} But it is ',
        ],
        "empty_prompt" : [
            f'{text} '
        ],
    },

    "SST2": lambda text :  {
        "dataset_prompt" : [
            f'{text}\nMy feedback to the film is ',
            f'{text}\nOverall, my feedback to the film is ',
            f'{text}\nAfter considering all aspects, my feedback to the film is ',
            f'{text}\nAfter considering all aspects, my viewpoint is ',
            f'{text}\nReflecting on the above, my viewpoint is ',
            f'{text}\nOverall, my perspective on the film is ',
            f'{text}\nOverall, my takeaway is ',
            f'{text}\nIn summary, I would say ',
            f'{text}\nI think it is ',
            f'{text}\nOverall, I think it is ',
            f'{text}\nConsidering everything, my feedback is',
            f'{text}\nConsidering everything, I think it is ',
            f'{text}\nAfter thinking about it, my feedback is ',
            f'{text}\nOverall, I see it as ',
            f'{text}\nTaking all factors into account, my assessment of it is ',
            
            f'{text}\nConsidering the details provided, my emotional reaction is ',
            f'{text}\nTaking into account the experience shared, my viewpoint is ',
            f'{text}\nReflecting on the content, my emotional stance is ',
            f'{text}\nGiven the information above, my sentiment evaluation is ',
            f'{text}\nAnalyzing the feedback, my emotional assessment is ',
            f'{text}\nBased on the review, my overall sentiment impression is ',
            f'{text}\nWeighing up the insights, my sentiment conclusion is ',
            f'{text}\nAfter thoroughly considering the review, my sentiment perspective is ',
        ],
        "question_prompt" : [
            f"Q: Here is a film review:{text} \nHelp me to detemine whether it is positive or negative?\nA:",

            f"Q: Given an film review: {text} \nIs the sentiment of this review positive or negative?\nA:",
            f"Q: Please analyze the following film review: {text} \nBased on the content and tone of the comment, do you think the reviewer's attitude towards the movie positive or negative?\nA:", #!new
            
        ],
        "ICL_prompt" : [
            f'Text: {text}\nSentiment:',
            f'Text: {text}\nSentiment Analysis: The overall sentiment is ',
        ],
        "sum_prompt" : [
            f'{text}\nAll in all, the film was ',
            f'{text}\nIn summary, the film was ',
            f'{text}\nIn essence, the film was ',
            f'{text}\nIn conclusion, the film was ',
            f'{text}\nTo sum up, the film was ',
        ],
        "short_prompt" : [
            f'{text} All in all ',
            f'{text} Just ',
            f'{text} It was ',
            f'{text} It is ',
            f'{text} That is ',
            f'{text} That\'s ',
            f'{text} But it is ',
        ],
        "empty_prompt" : [
            f'{text} '
        ],
    },
    
    "ag_news": lambda title, text: {
        "dataset_prompt" : [
            f'{title}\n{text}\nThis topic is about ',
            f'{title}\n{text}\nThe label that best describes this news article is ',
            f'{title}\n{text}\nThis piece of news is regarding ',

            f'{title}\n{text}\nThe news article is about ',
            f'{title}\n{text}\nCentral themes of this news piece encompass ',
            f'{title}\n{text}\nThe central theme of this article revolves around ',

            f'{title}\n{text}\nIt can be labeled as ',
            f'{title}\n{text}\nIts category is ',
            f'{title}\n{text}\nIn this article, it talks about ',
            f'{title}\n{text}\nThe content is a kind of ',
            f'{title}\n{text}\nI think the news can be classified as ',
            f'{title}\n{text}\nI would classify it as ',
            f'{title}\n{text}\nBased on the discription, its category is ',
            f'{title}\n{text}\nIn this context, the content falls into the category of ',
        ],
        "question_prompt" : [
            f"Q: Given an AG News article: {title}\n{text} \nWhich of the four standard categories (Business, Science/Technology, Sports, World/Politics) does this article best fit into?\nA:",
            f"Q: Review this news: {title}\n{text} \nIn which section (Business, Science/Tech, Sports, World/Politics) would you expect to find this article?\nA:",
            
            f"Q: Please analyze the article {title}\n{text} \nHelp me to determine its main category (Business, Science/Tech, Sports, World/Politics).\nA:", 
            
        ],
        "ICL_prompt" : [
            f'Text: {title} {text}\nCategory:',
            f'Text: {title} {text}\nTopic Classification: The overall topic is ',
        ],
        "sum_prompt" : [
            f'{title}\n{text}\nAll in all, it was ',
            f'{title}\n{text}\nIn summary, it was ',
            f'{title}\n{text}\nIn essence, it was ',
            f'{title}\n{text}\nIn conclusion, it was ',
            f'{title}\n{text}\nTo sum up, it\'s ',
        ],
        "short_prompt" : [
            f'{title}\n{text} All in all ',
            f'{title}\n{text} Just ',
            f'{title}\n{text} It was ',
            f'{title}\n{text} It is ',
            f'{title}\n{text} That is ',
            f'{title}\n{text} That\'s ',
            f'{title}\n{text} But it is ',
        ],
        "empty_prompt" : [
            f'{title}\n{text} '
        ],
    },

    "dbpedia": lambda title, text: {
        "dataset_prompt" : [
            #f'{title}\n{text}\nThe category of {title} is ',
            f'{title}\n{text}\nThe label that best describes {title} is ',
            f'{title}\n{text}\nSo, {title} is ',

            f'{title}\n{text}\nIn this sentence, {title} is ',
            f'{title}\n{text}\n{title} is a kind of ',
            f'{title}\n{text}\n{title} can be classified as ',
            f'{title}\n{text}\n{title} is an example of ',
            
            f'{title}\n{text}\n{title} belongs to ',
            #f'{title}\n{text}\nI think {title} is ',
            f'{title}\n{text}\nI would classify {title} as ',
            f'{title}\n{text}\nBased on the discription, its category is ',
            f'{title}\n{text}\nIn this context, {title} falls into the category of ',

            
            f'{title}\n{text}\nThe category of it is ',
            f'{title}\n{text}\nThe label that best describes it is ',
            f'{title}\n{text}\nSo, it is ',

            f'{title}\n{text}\nIn this sentence, it is ',
            f'{title}\n{text}\nIt is a kind of ',
            f'{title}\n{text}\nIt can be classified as ',
            f'{title}\n{text}\nIt is an example of ',
            
            f'{title}\n{text}\nIt belongs to ',
            f'{title}\n{text}\nI think it is ',
            f'{title}\n{text}\nI would classify it as ',
            f'{title}\n{text}\nIn this context, it falls into the category of ',
        ],
        "question_prompt" : [
            f"Q: Please examine the details below:\n{title}\n{text}\nIdentify the most suitable category or section for this entity from the following options: 'Album', 'Plant', 'WrittenWork', 'Film', 'EducationalInstitution', 'Building', 'MeanOfTransportation', 'Athlete', 'OfficeHolder', 'Company', 'NaturalPlace', 'Artist', 'Village', 'Animal'.\nA:",
            f"Q: Given the title and description: {title}\n{text} \nWhat category best fits the described entity? Choose from 'Album', 'Plant', 'WrittenWork', 'Film', 'EducationalInstitution', 'Building', 'MeanOfTransportation', 'Athlete', 'OfficeHolder', 'Company', 'NaturalPlace', 'Artist', 'Village', 'Animal'.\nA:",
            f"Q: Based on the title: {title} and the following description: {text}, how would you define or categorize {title}? Select the best option from 'Album', 'Plant', 'WrittenWork', 'Film', 'EducationalInstitution', 'Building', 'MeanOfTransportation', 'Athlete', 'OfficeHolder', 'Company', 'NaturalPlace', 'Artist', 'Village', 'Animal'.\nA:",
            
        ],
        "ICL_prompt" : [
            f'Text: {title} {text}\nCategory:',
            f'Text: {title} {text}\nTopic Classification: The overall topic is ',
        ],
        "sum_prompt" : [
            f'{title}\n{text}\nAll in all, it is ',
            f'{title}\n{text}\nIn summary, it is ',
            f'{title}\n{text}\nIn essence, it is ',
            f'{title}\n{text}\nIn conclusion, it is ',
            f'{title}\n{text}\nTo sum up, it\'s ',
        ],
        "short_prompt" : [
            f'{title}\n{text} All in all ',
            f'{title}\n{text} Just ',
            f'{title}\n{text} It was ',
            f'{title}\n{text} It is ',
            f'{title}\n{text} That is ',
            f'{title}\n{text} That\'s ',
            f'{title}\n{text} But it is ',
        ],
        "empty_prompt" : [
            f'{title}\n{text} '
        ],
    },

    "yahoo_answers": lambda title, text: {
        "dataset_prompt" : [
            f'{title}\n{text}\nThis topic is about ',
            f'{title}\n{text}\nThe label that best describes this question is ',

            f'{title}\n{text}\nThis issue is regarding ',
            f'{title}\n{text}\nThis discussion is about ',
            f'{title}\n{text}\nThis discussion is regarding ',
            f'{title}\n{text}\nThis issue is about ',
            f'{title}\n{text}\nThe label that best describes this issue is ',
            
            f'{title}\n{text}\nI would classify this question as ',
            f'{title}\n{text}\nIt can be labeled as ',
            f'{title}\n{text}\nOverall, The most fitting category for this issue is ',
            f'{title}\n{text}\nThe content is associated with ',
            f'{title}\n{text}\nI think it belongs to ',
            f'{title}\n{text}\nI would classify it as ',
            f'{title}\n{text}\nThis issue falls into the category of ',
        ],
        "question_prompt" : [
            f"Q: Given a Yahoo Answers thread: {title}\n{text} \nWhat category would you classify this question under? Choose from: 'Politics & Government', 'Society & Culture', 'Entertainment & Music', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Family & Relationships', 'Sports', 'Business & Finance'.\nA:",
            f"Q: Please analyze the following question and its response from Yahoo Answers: {title}\n{text} \nBased on its content and context, into which category or subject of Yahoo Answers would you place this question and answer? Select from: 'Politics & Government', 'Society & Culture', 'Entertainment & Music', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Family & Relationships', 'Sports', 'Business & Finance'.\nA:",
            f"Q: Determine the category of the following inquiry on Yahoo Answers: {title}\n{text} \nWhat is the primary topic or subject area of this query? Pick one from: 'Politics & Government', 'Society & Culture', 'Entertainment & Music', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Family & Relationships', 'Sports', 'Business & Finance'.\nA:",
        ],
        "ICL_prompt" : [
            f'Text: {title} {text}\nCategory:',
            f'Text: {title} {text}\nTopic Classification: The overall topic is ',
        ],
        "sum_prompt" : [
            f'{title}\n{text}\nAll in all, it was ',
            f'{title}\n{text}\nIn summary, it was ',
            f'{title}\n{text}\nIn essence, it was ',
            f'{title}\n{text}\nIn conclusion, it was ',
            f'{title}\n{text}\nTo sum up, it\'s ',
        ],
        "short_prompt" : [
            f'{title}\n{text} All in all ',
            f'{title}\n{text} Just ',
            f'{title}\n{text} It was ',
            f'{title}\n{text} It is ',
            f'{title}\n{text} That is ',
            f'{title}\n{text} That\'s ',
            f'{title}\n{text} But it is ',
        ],
        "empty_prompt" : [
            f'{title}\n{text} '
        ],
    },
    
    "isear": lambda text :  {
        "dataset_prompt" : [
            f'{text}\nIn summary, I would say ',
            f'{text}\nI think it is ',
            f'{text}\nOverall, I think it is ',
            f'{text}\nConsidering everything, I think it is ',
            f'{text}\nOverall, I see it as ',
            f'{text}\nIn summary, I would say ',
            f'{text}\nI feel ',
            f'{text}\nOverall, I feel ',
            f'{text}\nOverall, my feeling towards it is ',
            f'{text}\nThis text expresses ',
            f'{text}\nIt is a feeling of ',
            f'{text}\nThe sentiment is ',
            f'{text}\nIt is ',
            f'{text}\nThis conveys a sense of ',
            f'{text}\nI am ',
            f'{text}\nThe overall impression is ',
            f'{text}\nFrom my perspective, it is ',
            f'{text}\nIn my view, the feeling is ',

            f'{text}\nThis passage makes me feel ',
            f'{text}\nIt seems to evoke a feeling of ',
            f'{text}\nThis text primarily conveys ',
            f'{text}\nFrom this, I sense an emotion of ',
            f'{text}\nIt can be interpreted as expressing ',
            f'{text}\nThe underlying emotion seems to be ',
            f'{text}\nThis narrative elicits ',
            f'{text}\nFeeling-wise, this comes across as ',
            f'{text}\nThis evokes ',
            f'{text}\nThe emotional tone here is ',
            f'{text}\nThis story is imbued with ',
            f'{text}\nThe mood conveyed here is ',
            f'{text}\nReading this, one might feel ',
            f'{text}\nThis piece stirs up ',
            f'{text}\nThe primary sentiment here is ',
            f'{text}\nThis text is charged with ',
            f'{text}\nEmotionally, this is ',
            f'{text}\nThis depicts ',
            f'{text}\nOne could interpret this as ',
            f'{text}\nThis text leaves the impression of ',
        ],
        "question_prompt" : [
            f"Q: Given the following personal reflection: {text} \nWhich emotion does this reflection most strongly convey: sadness, joy, love, anger, fear or surprise?\nA:",
            f"Q: Please analyze the emotional tone of this statement: {text} \nBased on the language and sentiment, what primary emotion is being expressed: sadness, joy, love, anger, fear or surprise?\nA:",
            f"Q: Interpret the emotional expression in the given narrative: {text} \nConsidering the language and tone, what is the dominant emotion? Choose from: sadness, joy, love, anger, fear, surprise.\nA:",
           
        ],
        "ICL_prompt" : [
            f'Text: {text}\nEmotion:',
            f'Text: {text}\nEmotion Recognition: The overall emotion is ',
        ],
        "sum_prompt" : [
            f'{text}\nAll in all, it was ',
            f'{text}\nIn summary, it was ',
            f'{text}\nIn essence, it was ',
            f'{text}\nIn conclusion, it was ',
            f'{text}\nTo sum up, it was ',
        ],
        "short_prompt" : [
            f'{text} All in all ',
            f'{text} Just ',
            f'{text} It was ',
            f'{text} It is ',
            f'{text} That is ',
            f'{text} That\'s ',
            f'{text} But it is ',
        ],
        "empty_prompt" : [
            f'{text} '
        ],
    },

    "carer": lambda text :  {
        "dataset_prompt" : [
            f'{text}\nIn summary, I would say ',
            f'{text}\nI think it is ',
            f'{text}\nOverall, I think it is ',
            f'{text}\nConsidering everything, I think it is ',
            f'{text}\nOverall, I see it as ',
            f'{text}\nIn summary, I would say ',
            f'{text}\nI feel ',
            f'{text}\nOverall, I feel ',
            f'{text}\nOverall, my feeling towards it is ',
            f'{text}\nThis text expresses ',
            f'{text}\nIt is a feeling of ',
            f'{text}\nThe sentiment is ',
            f'{text}\nIt is ',
            f'{text}\nThis conveys a sense of ',
            f'{text}\nI am ',
            f'{text}\nThe overall impression is ',
            f'{text}\nFrom my perspective, it is ',
            f'{text}\nIn my view, the feeling is ',

            f'{text}\nThis passage makes me feel ',
            f'{text}\nIt seems to evoke a feeling of ',
            f'{text}\nThis text primarily conveys ',
            f'{text}\nFrom this, I sense an emotion of ',
            f'{text}\nIt can be interpreted as expressing ',
            f'{text}\nThe underlying emotion seems to be ',
            f'{text}\nThis narrative elicits ',
            f'{text}\nFeeling-wise, this comes across as ',
            f'{text}\nThis evokes ',
            f'{text}\nThe emotional tone here is ',
            f'{text}\nThis story is imbued with ',
            f'{text}\nThe mood conveyed here is ',
            f'{text}\nReading this, one might feel ',
            f'{text}\nThis piece stirs up ',
            f'{text}\nThe primary sentiment here is ',
            f'{text}\nThis text is charged with ',
            f'{text}\nEmotionally, this is ',
            f'{text}\nThis depicts ',
            f'{text}\nOne could interpret this as ',
            f'{text}\nThis text leaves the impression of ',
        ],
        "question_prompt" : [
            f"Q: Given the following personal reflection: {text} \nWhich emotion does this reflection most strongly convey: sadness, joy, love, anger, fear or surprise?\nA:",
            f"Q: Please analyze the emotional tone of this statement: {text} \nBased on the language and sentiment, what primary emotion is being expressed: sadness, joy, love, anger, fear or surprise?\nA:",
            f"Q: Interpret the emotional expression in the given narrative: {text} \nConsidering the language and tone, what is the dominant emotion? Choose from: sadness, joy, love, anger, fear, surprise.\nA:",
           
        ],
        "ICL_prompt" : [
            f'Text: {text}\nEmotion:',
            f'Text: {text}\nEmotion Recognition: The overall emotion is ',
        ],
        "sum_prompt" : [
            f'{text}\nAll in all, it was ',
            f'{text}\nIn summary, it was ',
            f'{text}\nIn essence, it was ',
            f'{text}\nIn conclusion, it was ',
            f'{text}\nTo sum up, it was ',
        ],
        "short_prompt" : [
            f'{text} All in all ',
            f'{text} Just ',
            f'{text} It was ',
            f'{text} It is ',
            f'{text} That is ',
            f'{text} That\'s ',
            f'{text} But it is ',
        ],
        "empty_prompt" : [
            f'{text} '
        ],
    },

}

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def get_input(dataset_name , sample):
    if dataset_name == "amazon_review_polarity":
        prompt_list = prompt_templates[dataset_name]( 
            title=sample[2] , text=sample[3] 
        )
    if dataset_name == "aclImdb":
        prompt_list = prompt_templates[dataset_name]( 
            text=sample[1]
        )
    if dataset_name == "ag_news":
        prompt_list = prompt_templates[dataset_name]( 
            title=sample[2] , text=sample[3] 
        )
    if dataset_name == "dbpedia":
        prompt_list = prompt_templates[dataset_name]( 
            title=sample[2] , text=sample[3] 
        )
    if dataset_name == "yahoo_answers":
        prompt_list = prompt_templates[dataset_name]( 
            title=sample[2]+str(sample[3]) , text=sample[4]
        )
    if dataset_name == "SST2":
        prompt_list = prompt_templates[dataset_name]( 
            text=sample[2] 
        )
    if dataset_name == "isear":
        prompt_list = prompt_templates[dataset_name]( 
            text=sample[2] 
        )
    if dataset_name == "carer":
        prompt_list = prompt_templates[dataset_name]( 
            text=sample[2] 
        )

    return prompt_list
