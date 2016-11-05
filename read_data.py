import xml.etree.ElementTree as ET
import re
import os
import pickle
import sys

dir = '/scratch/ceb545/nlp/project/who_did_what/Strict/'
#fi = 'train.xml'
fi = sys.argv[1]

passage_out = 'context.txt'
question_out = 'qu.txt'
choice_out = 'choices.txt'
labels_out = 'labels.txt'
category = fi.split('.')[0]

file_list = [passage_out,question_out,choice_out,labels_out]

def create_dirs():
    # create dir to save files if doesnt exist
    savedir = os.path.join(dir,category)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

def open_fi():
    p = open(os.path.join(dir,category,passage_out),'w')
    q = open(os.path.join(dir,category,question_out),'w')
    c = open(os.path.join(dir,category,choice_out),'w')
    l = open(os.path.join(dir,category,labels_out),'w')
    return p,q,c,l

def parse_xml(path):
    parser = ET.XMLParser()
    parser.entity["&nbsp;"] = unichr(160)
    #tree = parser.parse(path)
    entites = [('&nbsp;',u'\u00a0')]
    data_str = open(path).read()
    for before,after in entites:
        data_str = data_str.replace(before, after.encode('utf8'))
    root = ET.fromstring(data_str)
    return root

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\n"," ",string)
    return string.strip().lower()

def process_question(question_child):
    """
    process question: join left and right context
    """
    question_str = ""
    for child in question_child:
        if ((child.tag =='leftcontext')|(child.tag=='rightcontext')):
            if child.text!=None:
                if child.tag=='rightcontext':
                    question_str += 'xxx'
                question_str += child.text
    return clean_str(question_str)

def rootstuff(root, pa_fi,q_fi,ch_fi,lab_fi): 
    writefile = None
    for child in root:
        choices = []
        for subchild in child:
            child_text = clean_str(subchild.text).encode('utf-8')
            if subchild.tag =='question':
                line = process_question(subchild).encode('utf-8')
                q_fi.write("%s\n"%line)
            if subchild.tag == 'contextart':
                pa_fi.write("%s\n"%child_text)
            if subchild.tag == 'choice':
                if subchild.attrib['correct']=='true':
                    lab_fi.write("%s\n"%child_text)
                choices.append(child_text)
        ch_fi.write("%s\n"%('$$$'.join(choices)))

if __name__=='__main__':
    file = sys.argv[1]
    create_dirs()
    pa_fi,q_fi,ch_fi,lab_fi = open_fi()


    data_root = parse_xml(os.path.join(dir,fi))
    rootstuff(data_root,pa_fi,q_fi,ch_fi,lab_fi)

    pa_fi.close()
    q_fi.close()
    ch_fi.close()
    lab_fi.close()
