from nltk import word_tokenize
from  tqdm import tqdm
import spacy
from ipdb import set_trace
import argparse
import os




def handle_ent(doc,e):
    token_text = []
    flag1=False
    left_neighbor = None
    right_neighbor = None
    site=-1
    for token in doc:
        if token.text == e or e in token.text:
            # left_neighbor = token.nbor(-1) if token.i > 0 else None
            # right_neighbor = token.nbor(1) if token.i < len(doc) - 1 else None
            # if left_neighbor!=None :
            #     token_text.append(left_neighbor.text)
            # if right_neighbor!=None:
            #     token_text.append(right_neighbor.text)
            if token.head.text in token_text:
                continue
            token_text.append(token.head.text)
            return token_text
    div_text = word_tokenize(e)
    if 'N(5)-' in e and 'H(4)' in e:
        site = e.find('H(4)')
        div_text = [e[:site-1],e[site:]]
    if e =='[5-14C]DFMO':
        div_text = ['[5','-','14C]DFMO']

    for token in doc:
        if token.text in div_text or token.text in div_text[0]:

            # left_neighbor = token.nbor(-1) if token.i > 0 else None
            # right_neighbor = token.nbor(1) if token.i < len(doc) - 1 else None
            # if left_neighbor!=None:
            #     token_text.append(left_neighbor.text)
            # if right_neighbor!=None:
            #     token_text.append(right_neighbor.text)
            if token.head.text in token_text:
                continue
            token_text.append(token.head.text)
            # return token_text
    
    return token_text

def dropout_title(text):
    sentence = text
    all_title = ['[s1]','[e1]','[s2]','[e2]']
    for title in all_title:
        site = sentence.find(title)
        if site ==-1:
            set_trace()
        sentence = sentence[:site]+sentence[site+len(title):]
    return sentence
def find_shortest_distance_and_path(tree, root):
    target1 = 'e1'
    target2 = 'e2'
    if not root:
        return None, [] 


    has_found = []
    queue = [(root, [root])]
    found1 = False
    found2 = False
    path1 = []
    path2 = []
    while queue:
        node, path = queue.pop(0)
        if node in has_found:
            continue
        else:
            has_found.append(node)

        if node['ent_type'] == target1:
            found1 = True
            path1 = path


        if node['ent_type'] == target2:
            found2 = True
            path2 = path


        if found1 and found2:
            return path1, path2


        for child in node['children']:
            queue.append((tree[child], path + [tree[child]]))


    return path1,path2
def has_problem(sentences):
    site1 = sentences.find(' [s1]')
    site2 = sentences.find(' [s2]')
    site3 = sentences.find(' [e1]')
    site4 = sentences.find(' [e2]')
    if (site1 <site2 and site3>site2) or (site1>site2 and site4>site1):
        return True
    return False

def main(data,save_type):

    nlp = spacy.load("en_core_sci_lg")
    label_text_4 = []
    all_title = ['s1','e1','s2','e2']
    for id,sentences in tqdm(enumerate(data[:]),total=len(data)):
        neg_sentence = []
        neg_child_sentence = []
        neg_child = []
        e1_neg = []
        e2_neg = []
        site2idx = {}
        sentence,e1,e2,e1_type,e2_type,rel,distance = sentences.split('\t')
        e1_child = {'text':e1,'idx':[],'ent_type':'e1','children':[],'head':[]}
        e2_child = {'text':e2,'idx':[],'ent_type':'e2','children':[],'head':[]}
        flag = has_problem(sentence)
        rel=rel.strip()
        raw_text = dropout_title(sentence)
        raw_text = " ".join(word_tokenize(raw_text))
        doc1 = nlp(sentence)
        doc2 = nlp(raw_text)
        for token in doc1:
            neg_sentence.append(token.text)
        i=0
        for token in doc2:
            neg_child_sentence.append({'text':token.text,'idx':[token.idx],'ent_type':None,'children':[child.idx for child in token.children],'head':token.head.text})
            site2idx[token.idx] = i
            i+=1
        a = []
        b = []
        for (i,temp) in enumerate(neg_child_sentence):
            a.extend(temp['idx'])
            b.extend(temp['children'])
        root_site = []
        for i in a:
            if i not in b:
                root_site.append(site2idx[i])
        neg_label=''
        i=0
        j=0
        while i <len(neg_sentence):
            if neg_sentence[i].strip()=='':
                i+=1
            elif neg_sentence[i]=='[':
                if neg_sentence[i+1] in all_title and neg_sentence[i+2]==']':
                    neg_child_sentence.insert(j,' ['+neg_sentence[i+1]+']')
                    i+=3
                    j+=1
                else:
                    i+=1
                    j+=1
            else:
                i+=1
                j+=1
        flag1 = False
        flag2 = False
        for (i,neg_sent) in enumerate(neg_child_sentence):
            if neg_sent == ' [s1]':
                flag1 = True
                site1 = i+1
            elif neg_sent == ' [s2]':
                flag2 = True
                site2 = i +1
            elif neg_sent ==' [e1]':
                flag1 =False
                site3 = i-1
            elif neg_sent ==' [e2]':
                flag2 =False
                site4 = i-1
            else:
                if flag1 ==True:
                    neg_child_sentence[i]['ent_type'] = 'e1'
                    e1_child['idx'].extend(neg_sent['idx'])
                    e1_child['head'].append(neg_sent['head'])
                    for site in neg_sent['children']:
                        if site not in e1_child['children'] and site not in e1_child['idx']:
                            e1_child['children'].append(site2idx[site])
                elif flag2 ==True:
                    neg_child_sentence[i]['ent_type'] = 'e2'
                    e2_child['idx'].extend(neg_sent['idx'])
                    e2_child['head'].append(neg_sent['head'])
                    for site in neg_sent['children']:
                        if site not in e2_child['children'] and site not in e2_child['idx']:
                            e2_child['children'].append(site2idx[site])

        for i in range(site1,site3+1):
            neg_child_sentence[i] = e1_child
            neg_child_sentence[i]['site'] = i
        for i in range(site2,site4+1):
            neg_child_sentence[i] = e2_child
            neg_child_sentence[i]['site'] = i
        path1 = []
        path2 = []
        if flag==False:
            for temp in [' [s1]',' [s2]',' [e1]',' [e2]']:
                neg_child_sentence.remove(temp)
            for (i,temp) in enumerate(neg_child_sentence):
                if temp['ent_type'] == 'e1' or temp['ent_type'] =='e2':
                    continue
                for (j,child_idx) in enumerate(temp['children']):
                    neg_child_sentence[i]['children'][j] = site2idx[child_idx]

            for i in root_site:
                root = neg_child_sentence[i]
                path1,path2 = find_shortest_distance_and_path(neg_child_sentence,root)
                if len(path1)!=0 and len(path2)!=0:
                    break

        len1 = len(path1)
        len2 = len(path2)
        length = len1 if len1<len2 else len2
        if len1==0 or len2==0:
            e1_neg.extend(e1_child['head'])
            e2_neg.extend(e2_child['head'])
        else:
            for i in range(length):
                if path1[i] != path2[i]:
                    neg_child.extend(path1[i-1:-1])
                    neg_child.extend(path2[i:-1])
                    break
            for i in range(len(neg_child)):
                neg_child[i] = neg_child[i]['text']

            if len(neg_child)==0:
                if len1+1 == len2:
                    neg_child.extend(e1_child['head'])
                elif len2+1 == len1:
                    neg_child.extend(e2_child['head'])
                elif len1 + 1 < len2:
                    neg_child.extend(path2[len1:-1])
                    for i in range(len(neg_child)):
                        neg_child[i] = neg_child[i]['text']
                elif len2 + 1 <len1:
                    neg_child.extend(path1[len2:-1])
                    for i in range(len(neg_child)):
                        neg_child[i] = neg_child[i]['text']
        e1_neg.extend(neg_child)
        e2_neg.extend(neg_child)
        if len(e1_neg) ==0 or len(e2_neg)==0:
            print(id)
        label_text = [sentence,e1_type,e2_type,e1,e2,rel,' '.join(e1_neg),' '.join(e2_neg)+'\n']
        label_text_4.append('\t'.join(label_text))
    os.makedirs("neg_mid_dataset/{}/".format(save_type),exist_ok=True)
    with open('neg_mid_dataset/{}/normal_{}.txt'.format(save_type,save_type),'w') as f:
        f.writelines(''.join(label_text_4))
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    args = parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    for flag in ["train","dev"]:
        with open('{}/{}.txt'.format(args.data_path,flag),'r') as f:
            data = f.readlines()
        main(data,flag)
