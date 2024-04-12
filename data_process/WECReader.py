"""

Usage:
	WECReader.py --train==<File> --dev==<File> --test==<test> --dataset=<dataset> [--ratio=<x>] [--topic=<type>]

Options:
	-h --help     Show this screen.
	--dataset=<dataset>   wec/ecb - which dataset to generate for [default: wec]
	--split=<set>    dev/test/train/na (split=na => doesnt matter) [default: na].
	--ratio=<x>  ratio of positive:negative, were negative is the controlled list (ratio=-1 => no ratio) [default: -1]
	--topic=<type>  subtopic/topic/corpus - relevant only to ECB+, take pairs only from the same sub-topic, topic or corpus wide [default: corpus]

"""
# 为原始数据集添加参数信息

import os,sys
from allennlp.predictors.predictor import Predictor
import json
import spacy

from os import path

from docopt import docopt
from tqdm import tqdm


# data = {}
# srl_result = {}
# pos_result = {}

# event_all = []
# entity_all = []

# validate_sen = {}

import spacy
from allennlp.predictors.predictor import Predictor





class Entity():
	def __init__(self,text,type,start,end) -> None:
		self.text = text
		self.type = type
		self.start_offset=start
		self.end_offset = end

def dataset_process(file_path):
	""" 读入原始数据加入参数信息 """
	
	data_feat = load_json_file(file_path)
	for mention in tqdm(data_feat,desc="mention"):

		sentence = " ".join(token for token in mention['mention_context']).strip()
		event_start_offset,event_end_offset=mention['tokens_number'][0],mention['tokens_number'][-1]
		doc = nlp(sentence)

		try:
			entities=Get_entities(doc,mention['mention_context'])
			# Palo Alto-based 标注的是palo alto  但我应该在这个将实体定义为 palo alto-based，这样在tokenizer时也能拼起来
			src_result = predictor.predict(sentence=sentence)
					
			event_arg0,event_arg1,event_loc,event_time =Get_arguments(mention['mention_context'],src_result,event_start_offset,event_end_offset,entities)
			# print(event_arg0,event_arg1,event_loc,event_time)
		except Exception as e:
				print("error:",e)
		arguments=[]
		arguments.append([event_arg0[0],event_arg0[1],mention['mention_context'][event_arg0[0]:event_arg0[1]+1] ])
		arguments.append([event_arg1[0],event_arg1[1],mention['mention_context'][event_arg1[0]:event_arg1[1]+1] ])
		arguments.append([ event_loc[0],event_loc[1],mention['mention_context'][event_loc[0]:event_loc[1]+1]])
		arguments.append([event_time[0],event_time[1],mention['mention_context'][event_time[0]:event_time[1]+1] ])
		mention['arguments']=arguments
	write_arguments(data_feat,file_path)
# ../datasets/WEC-Eng/Train_Event_gold_mentions.json
def write_arguments(data,filepath):
	basename = path.basename(path.splitext(filepath)[0])
	dirname = os.path.dirname(filepath)
	arguments_file = dirname + "/arguments/" + basename + ".json" #样本比例为-1，即默认
	with open(arguments_file,'w',encoding='utf-8')as f:
		json.dump(data,f,ensure_ascii=False)
		f.close()
def Get_arguments(tokens,srl_dict,event_start_offset,event_end_offset,entities):
	idx_mapping = {}
	i,j = 0,0
	srl_tokens = srl_dict['words']
	while i < len(tokens):
		token = tokens[i]
		while True:
			if token.startswith(srl_tokens[j]):
				start_idx = j
				token_srl = srl_tokens[j]
				j += 1
				break
			else:
				j += 1
		while token != token_srl and j < len(srl_tokens):
			if token.startswith((token_srl+srl_tokens[j])):
				token_srl += srl_tokens[j]
			j += 1
		idx_mapping[i] = (start_idx, j)#空格分的token与预测器分的token的对应关系
		i += 1
	arg0,arg1,loc,time=(-1,-1),(-1,-1),(-1,-1),(-1,-1),

	#这里是ecb的逻辑，触发词作为事件mention,但是wec应该怎么标，只是一个事件mention，参数信息大多并不是描述这个时间的
	for verb in srl_dict['verbs']:
		# flag = False
		# for i in range(idx_mapping[event_start_offset][0], idx_mapping[event_end_offset][1]):

			# if '-V' in verb['tags'][i]:  #
			#     flag = True
			#     break
		# if not flag:
		#     continue

		for entity in entities:
			try:	
				for i in range(idx_mapping[entity.start_offset][0], idx_mapping[entity.end_offset][1]):
					if verb['tags'][i] != 'O':
						type = verb['tags'][i].split('-')[-1]
						if type == 'ARG0' and 'GPE' not in entity.type and 'DATE' not in entity.type:
							arg0 = (entity.start_offset, entity.end_offset)
							break
						if type == 'ARG1' and 'GPE' not in entity.type and 'DATE' not in entity.type:
							arg1 = (entity.start_offset, entity.end_offset)
							break
						if type == 'GPE' and 'GPE' in entity.type:
							loc = (entity.start_offset, entity.end_offset)
							break
						if type == 'TMP' and 'DATE' in entity.type:
							time = (entity.start_offset, entity.end_offset)
							break
			except Exception as e:
				print(e)
				
	return arg0,arg1,loc,time
def Get_entities(doc,tokens):
	i=0
	entities=[]
	for entity in doc.ents:#entity是按顺序给出的
		start,end=-1,-1
		flag = True
		while(i<len(tokens) and flag):
			#已token开头则说明是一个独立的单词,如as 与a就不匹配，就不会错误
			if (entity.text.startswith(tokens[i])and entity.text.split(' ')[0]==tokens[i]) or tokens[i].startswith(entity.text.split(' ')[-1]):
				start = i
				# palo alto ->palo alto-based
				while True:
					if (entity.text.endswith(tokens[i]) and entity.text.split(' ')[-1]==tokens[i]) or tokens[i].startswith(entity.text.split(' ')[-1]):
						end = i
						flag=False#标记当前entity已找到
						break
					i+=1    
			i+=1
		entities.append(Entity(entity.text,entity.label_,start,end)) #不能用这个start，end 并不是按空格划分的
	return entities
def wec_process(file_list):
	for file in file_list:
		dataset_process(file)
def load_json_file(file_path):
	"""load a file into a json object"""
	try:
		with open(file_path, encoding='utf-8') as small_file:
			return json.load(small_file)
	except OSError as e:
		print(e)
		print('trying to read file in blocks')
		with open(file_path, encoding='utf-8') as big_file:
			json_string = ''
			while True:
				block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
				json_string = json_string + block
				if not block:  # Reached EOF
					break
			return json.loads(json_string)
if __name__ == '__main__':
	os.chdir(sys.path[0])
	# --train==<File> --dev==<File> --test==<test> --dataset=<dataset> [--ratio=<x>] [--topic=<type>]
	argv=['--train','../datasets/WEC-Eng/Train_Event_gold_mentions.json','--dev','../datasets/WEC-Eng/Dev_Event_gold_mentions_validated.json','--test','../datasets/WEC-Eng/Test_Event_gold_mentions_validated.json','--dataset','wec','--ratio',10]
	arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
	print(arguments)
	train_file=arguments.get("--train")
	dev_file=arguments.get("--dev")
	test_file=arguments.get("--test")

	global nlp,predictor
	nlp= spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
	predictor=Predictor.from_path("../model/structured-prediction-srl-bert.2020.12.15.tar.gz")

	
	wec_process([train_file])


