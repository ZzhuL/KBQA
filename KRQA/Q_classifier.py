# -*- encoding: utf-8 -*-
"""
    @Project: KR_QA_Medical.py
    @File   : Q_classifier.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/13  9:22
"""
import os
import ahocorasick  # 调用这个库函数进行知识推理


class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 特征词路径
        self.disease_path = os.path.join(cur_dir, 'dict/disease.txt')  # 疾病
        self.department_path = os.path.join(cur_dir, 'dict/department.txt')  # 科室
        self.check_path = os.path.join(cur_dir, 'dict/check.txt')  # 检查
        self.drug_path = os.path.join(cur_dir, 'dict/drug.txt')  # 药物
        self.food_path = os.path.join(cur_dir, 'dict/food.txt')  # 食物
        self.producer_path = os.path.join(cur_dir, 'dict/producer.txt')  # 药品大类
        self.symptom_path = os.path.join(cur_dir, 'dict/symptom.txt')  # 症状
        # self.infection_path = os.path.join(cur_dir, 'dict/infection.txt')  # 传染方式
        self.deny_path = os.path.join(cur_dir, 'dict/deny.txt')  # 否认
        # self.expense_path = os.path.join(cur_dir, 'dict/expense.txt')  # 费用
        # 加载特征词
        self.disease_wds = [i.strip() for i in open(self.disease_path, encoding="utf-8") if i.strip()]
        self.department_wds = [i.strip() for i in open(self.department_path, encoding="utf-8") if i.strip()]
        self.check_wds = [i.strip() for i in open(self.check_path, encoding="utf-8") if i.strip()]
        self.drug_wds = [i.strip() for i in open(self.drug_path, encoding="utf-8") if i.strip()]
        self.food_wds = [i.strip() for i in open(self.food_path, encoding="utf-8") if i.strip()]
        self.producer_wds = [i.strip() for i in open(self.producer_path, encoding="utf-8") if i.strip()]
        self.symptom_wds = [i.strip() for i in open(self.symptom_path, encoding="utf-8") if i.strip()]
        # #
        # self.infection_wds = [i.strip() for i in open(self.infection_path, encoding="utf-8") if i.strip()]
        # self.expense_wds = [i.strip() for i in open(self.expense_path, encoding="utf-8") if i.strip()]
        self.region_words = set(self.department_wds + self.disease_wds + self.check_wds + self.drug_wds +
                                self.food_wds + self.producer_wds + self.symptom_wds)
        self.deny_words = [i.strip() for i in open(self.deny_path, encoding="utf-8") if i.strip()]
        # 构造领域actree，基于树匹配比关键词分割匹配更高效，ahocorasick是个现成的快速匹配函数
        self.region_tree = self.build_actree(list(self.region_words))  # 调用下面的build_actre函数
        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()  # 调用下面定义的build_wdtype_dict函数，构造词类型
        # 问句疑问词
        self.infection_qwds = ['传染', '传播', '传递', '传播方式', '人传人']
        self.expense_qwds = ['费用', '多少钱', '钱', '人民币', '元']
        self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
        self.cause_qwds = ['原因', '成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何',
                           '如何才会', '怎么才会', '会导致', '会造成']
        self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现',
                              '伴随发生', '伴随', '共现']
        self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜', '忌口', '补品', '保健品', '食谱',
                          '菜谱', '食用', '食物', '补品']
        self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止', '躲避', '逃避', '避开', '免得', '逃开', '避开',
                             '避掉', '躲开', '躲掉', '绕开', '怎样才能不', '怎么才能不', '咋样才能不', '咋才能不',
                             '如何才能不', '怎样才不', '怎么才不', '咋样才不', '咋才不', '如何才不',
                             '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                             '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年']
        self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治',
                             '怎么办', '咋办', '咋治']
        self.cureprob_qwds = ['治愈率', '多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治',
                              '可治', '可以治', '可以医']
        self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
        self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
        self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
        self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处',
                          '用途', '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']

        print('model init finished ......')

        return

    '''分类主函数'''
    def classify(self, question):
        data = {}
        medical_dict = self.check_medical(question)  # 调用下面定义的check_medical问句过滤函数
        if not medical_dict:
            if 'diseases_dict' in globals() and True:  # 判断是否是首次提问，若首次提问，则diseases_dict无值
                medical_dict = diseases_dict
            else:
                return {}

        data['args'] = medical_dict
        # print(data)
        # 收集问句当中所涉及到的实体类型
        types = []
        for type_ in medical_dict.values():
            types += type_

        question_types = []

        # 症状
        if self.check_words(self.symptom_qwds, question) and ('disease' in types):
            # self.symptom_qwds来自于init，查找self.symptom_qwds是否在question内
            question_type = 'disease_symptom'
            question_types.append(question_type)

        if self.check_words(self.symptom_qwds, question) and ('symptom' in types):  # check_words是下面定义的特征词分类函数
            question_type = 'symptom_disease'
            question_types.append(question_type)

        # 原因
        if self.check_words(self.cause_qwds, question) and ('disease' in types):
            question_type = 'disease_cause'
            question_types.append(question_type)
        # 并发症
        if self.check_words(self.acompany_qwds, question) and ('disease' in types):
            question_type = 'disease_acompany'
            question_types.append(question_type)

        # 推荐食品
        if self.check_words(self.food_qwds, question) and 'disease' in types:
            deny_status = self.check_words(self.deny_words, question)
            if deny_status:
                question_type = 'disease_not_food'
            else:
                question_type = 'disease_do_food'
            question_types.append(question_type)

        # 已知食物找疾病
        if self.check_words(self.food_qwds+self.cure_qwds, question) and 'food' in types:
            deny_status = self.check_words(self.deny_words, question)
            if deny_status:
                question_type = 'food_not_disease'
            else:
                question_type = 'food_do_disease'
            question_types.append(question_type)

        # 推荐药品
        if self.check_words(self.drug_qwds, question) and 'disease' in types:
            question_type = 'disease_drug'
            question_types.append(question_type)

        # 药品治啥病
        if self.check_words(self.cure_qwds, question) and 'drug' in types:
            question_type = 'drug_disease'
            question_types.append(question_type)

        # 疾病接受检查项目
        if self.check_words(self.check_qwds, question) and 'disease' in types:
            question_type = 'disease_check'
            question_types.append(question_type)

        # 已知检查项目查相应疾病
        if self.check_words(self.check_qwds+self.cure_qwds, question) and 'check' in types:
            question_type = 'check_disease'
            question_types.append(question_type)

        # 症状预防
        if self.check_words(self.prevent_qwds, question) and 'disease' in types:
            question_type = 'disease_prevent'
            question_types.append(question_type)

        # 疾病医疗周期
        if self.check_words(self.lasttime_qwds, question) and 'disease' in types:
            question_type = 'disease_lasttime'
            question_types.append(question_type)

        # 疾病治疗方式
        if self.check_words(self.cureway_qwds, question) and 'disease' in types:
            question_type = 'disease_cureway'
            question_types.append(question_type)

        # 疾病治愈可能性
        if self.check_words(self.cureprob_qwds, question) and 'disease' in types:
            question_type = 'disease_cureprob'
            question_types.append(question_type)

        # 传染方式
        if self.check_words(self.infection_qwds, question) and 'disease' in types:
            question_type = 'disease_infection'
            question_types.append(question_type)

        # 治疗费用
        if self.check_words(self.expense_qwds, question) and 'disease' in types:
            question_type = 'disease_expense'
            question_types.append(question_type)

        # 疾病易感染人群
        if self.check_words(self.easyget_qwds, question) and 'disease' in types:
            question_type = 'disease_easyget'
            question_types.append(question_type)

        # 什么病挂什么科室
        if self.check_words(self.belong_qwds, question) and 'disease' in types:
            question_type = 'disease_department'
            question_types.append(question_type)

        # # 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
        if question_types == [] and 'disease' in types:
            question_types = ['disease_desc']
        # 若没有查到相关的外部查询信息，那么则将该疾病的描述信息返回
        if question_types == [] and 'symptom' in types:
            question_types = ['symptom_disease']

        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''构造词对应的类型'''
    def build_wdtype_dict(self):
        wd_dict = dict()
        for wd in self.region_words:  # 找到用户输入的词是什么范围的，比如用户输入高血压，这个单词属于疾病？食物？科室？还是药物
            wd_dict[wd] = []
            if wd in self.disease_wds:
                wd_dict[wd].append('disease')
            if wd in self.department_wds:
                wd_dict[wd].append('department')  # 疾病归属科室
            if wd in self.check_wds:
                wd_dict[wd].append('check')
            if wd in self.drug_wds:
                wd_dict[wd].append('drug')
            if wd in self.food_wds:
                wd_dict[wd].append('food')
            if wd in self.symptom_wds:
                wd_dict[wd].append('symptom')
            if wd in self.producer_wds:
                wd_dict[wd].append('producer')
        return wd_dict

    '''构造actree，加速过滤'''
    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()  # 初始化trie树，ahocorasick 库 ac自动化 自动过滤违禁数据
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))   # 向trie树中添加单词
        actree.make_automaton()   # 将trie树转化为Aho-Corasick自动机
        return actree

    '''问句过滤'''
    def check_medical(self, question):
        region_wds = []
        for i in self.region_tree.iter(question):   # ahocorasick库 匹配问题  iter返回一个元组，i的形式如(3, (23192, '乙肝'))
            wd = i[1][1]  # 匹配到的词
            region_wds.append(wd)
        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1 != wd2:
                    stop_wds.append(wd1)   # stop_wds取重复的短的词，如region_wds=['乙肝', '肝硬化', '硬化']，则stop_wds=['硬化']
        final_wds = [i for i in region_wds if i not in stop_wds]     # final_wds取长词
        # print("final_wds:", final_wds)
        final_dict = {i: self.wdtype_dict.get(i) for i in final_wds}  # 来自于构造词典，# 获取词和词所对应的实体类型
        # 联系上一个实体
        global diseases_dict
        if final_dict:
            diseases_dict = final_dict
        # print("final_dict:", final_dict)
        return final_dict

    '''基于特征词进行分类'''
    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False


if __name__ == '__main__':
    handler = QuestionClassifier()
    while 1:
        question = input('input an question:')
        data = handler.classify(question)
        print(data)

