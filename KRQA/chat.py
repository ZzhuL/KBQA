# -*- encoding: utf-8 -*-
"""
    @Project: KR_QA_Medical.py
    @File   : chat.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/3  9:22
"""
from Q_classifier import *
from sklearn_Classification.clf_model import *
from TransH_search import *
import warnings

# 忽略UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning)


class ChatBotGraph:
    def __init__(self):
        self.CLF = CLFModel("./sklearn_Classification/model_file/")  # 意图识别
        self.classifier = QuestionClassifier()  # 提取实体和关系
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        tent = self.CLF.predict(sent)  # 预测用户输入信息的意图
        answer = random.choice(gossip_corpus.get(tent))  # 生成初始答复
        if tent != "diagnosis":  # 判断为闲聊
            return answer
        else:
            res_classify = self.classifier.classify(sent)  # 利用classify函数先对其进行分类
            if not res_classify:
                return answer  # 没有找到对应实体
            # print("classify结果:", res_classify)
            answer = self.searcher.search_main(res_classify)
            if not answer:
                answer = random.choice(gossip_corpus.get("dont_know"))

            return answer


if __name__ == '__main__':
    handler = ChatBotGraph()
    print('\033[1;30;47m小M:', random.choice(gossip_corpus.get("greet")), '\033[0m')
    while 1:
        try:
            question = input("咨询:")
            answer = handler.chat_main(question)
            answer = '小M:' + answer
            for i in range(0, len(answer), 65):
                print('\033[1;30;47m', answer[i:i+65].ljust(65), '\033[0m')
        except KeyboardInterrupt:
            print('\n小M:\033[1;30;47m', random.choice(gossip_corpus.get("goodbye")), '\033[0m')
            exit()
