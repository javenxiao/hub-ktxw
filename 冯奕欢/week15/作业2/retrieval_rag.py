from RetrievalAgent import RetrievalAgent

if __name__ == '__main__':
    rag = RetrievalAgent()
    question1 = "诺如病毒的传染期和观察期限是多久？"
    response1 = rag.query_qa(question1)
    print("问：", question1)
    print("答：", response1)
    question2 = "秋冬季应该如何预防传染病？"
    response2 = rag.query_qa(question2)
    print("问：", question2)
    print("答：", response2)