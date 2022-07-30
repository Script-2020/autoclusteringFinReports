import time
import os
import argparse
import tqdm
import json
import numpy as np
from pathlib import Path
import importlib
import timeit
import requests
import random
import re
import torch
import multiprocessing as mp
import ray
from torch.utils.data import Dataset

import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path 
import fitz
import pikepdf
from bs4 import BeautifulSoup
from ray.util.multiprocessing.pool import Pool

from Levenshtein import distance as lev

PRETRAINED_VOCAB_FILES_MAP = { 
        "fra": "https://huggingface.co/flaubert/flaubert_base_uncased/raw/main/vocab.json",
        "eng": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt", 
        "deu": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt", 
        "chi": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "fin": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt", 
    }

langs = {'fra':'fr','eng':'en','deu':'de'}

@ray.remote
def remote_apply_ocr_and_html(info):
    return apply_ocr_and_html(info)

def apply_ocr_and_html(info):
    file_list = info['data']
    lang_modules = info['modules']
    sample_response_list = {} 
    for i_file_doc, fileDoc in enumerate(file_list):
        pdf_file_path = fileDoc['datafile']
        split_type = fileDoc['split_type']
        detect_lang = fileDoc['detect_lang']
        page_start = fileDoc['page_start'] 
        page_end = fileDoc['page_end']
        lang = fileDoc['lang']
        is_encrypted = False
        
        sample_response_list[pdf_file_path] =  {'words':[], 'bbox':[], 'pages':[]}
    
        #using pdf2toimage to get images from pdf

        try:
            with pikepdf.open(pdf_file_path) as pdf:
                sample_response_list[pdf_file_path]['total_pages'] =  len(pdf.pages)
            images = convert_from_path(pdf_file_path, dpi=300,first_page=page_start+1, last_page=page_end)
            sample_response_list[pdf_file_path]['last_page'] = page_start + len(images) 
        except Exception as ex:
            print('Error while opening the file ', pdf_file_path , '\n', ex)
            continue
        read_pdf_file_path = pdf_file_path 
        #using fitz to read html to check if document is scanned or encoded, exception check if document is encrypted
        with fitz.open(pdf_file_path) as doc_fitz_temp:            
            try:
                page_fitz_temp = doc_fitz_temp[0]
                del page_fitz_temp
            except ValueError as ex:
                decripted__file_path = decryptPDF(pdf_file_path)
                print('Decrypted for working ', pdf_file_path)
                print(ex)
                is_encrypted = True
                if decripted__file_path is None:
                    print('ERROR. Nothing to decrypt, on ', pdf_file_path, ' ', ex) 
                    raise ex 
                with fitz.open(decripted__file_path) as doc_fitz_temp: 
                    try:
                        page_fitz_temp = doc_fitz_temp[0]
                        del page_fitz_temp
                        read_pdf_file_path = decripted__file_path
                    except Exception as ex2:
                        print('ERROR, decrypted but with errors,  on ', pdf_file_path, '  ', ex2) 
                        continue
        
        doc_fitz = fitz.open(read_pdf_file_path)
        
        #using pytesseract to extract text, bbox and orientation information
        
        for i_page, image in enumerate(images):
            page_list = []
            word_list = []
            width, height = image.size
            
            if  height > width:
                orientation = 'P'
            else:
                orientation = 'L'
            file_name =  Path(pdf_file_path).stem 
            file_name_page = Path(pdf_file_path).stem + '_' + str(i_page+page_start)
            res_rotation = getPageTextRotating(image, file_name_page , detect_lang=detect_lang,lang_modules=lang_modules)

            if res_rotation['rotation'] == 90 or res_rotation['rotation'] ==270:
                orientation = 'L'
            
            if res_rotation['status'] != 'Empty':
                oLanguage = res_rotation['lang']
                rotation = res_rotation['rotation']
                #TODO :add all the languages that were discovered
                ocr_df = pytesseract.image_to_data(res_rotation['image'], output_type='data.frame', config = '-l ' + oLanguage[0])  
                ocr_df = ocr_df.dropna().reset_index(drop=True)

                oImage = Path(pdf_file_path)
            
                float_cols = ocr_df.select_dtypes('float').columns
                ocr_df = ocr_df.dropna().reset_index(drop=True)
                ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
                ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
                ocr_df = ocr_df.dropna().reset_index(drop=True)

                data_columns = ocr_df[['left', 'top', 'width', 'height','block_num','line_num','text','level', 'word_num','conf']]
                
                data = []
                
                for idx, row in data_columns.iterrows():
                    x, y, w, h, block_num, line_num, text, level, word_num, conf= tuple(row) 
                    data.append([ block_num, line_num, text, x, y, x + w, y + h, w, h, level, word_num, conf])
                    #               0           1       2    3  4    5      6    7  8   9        10       11     
                
                if split_type == 'line':
                    blocks = {item_block[0]:{item_line[1]:0 for item_line in data if item_line[0]==item_block[0]} for item_block in data}  
                    text_lines = []
                    for block in blocks:
                        for line in blocks[block]:
                            text_data = [item for item in data if item[0]==block and item[1]==line] 
                            text_data.sort(key=lambda x:x[4], reverse=False)
                            word_in_text_indexes = {i:0 for i,item in enumerate(text_data)}
                            while 0 in word_in_text_indexes.values():
                                first_word = [index for index in word_in_text_indexes if word_in_text_indexes[index] ==0]
                                if len(first_word) >0:
                                    first_word = first_word[0]
                                    line_data = [(index,text_data[index]) for index in word_in_text_indexes if text_data[index][4]<text_data[first_word][6] and word_in_text_indexes[index]==0]
                                    text_data_line = [item[1] for item in line_data]
                                    index_data_line = [item[0] for item in line_data]
                                    text_data_line.sort(key=lambda x:x[3], reverse=False)
                                    
                                    try:
                                        text_ = ' '.join([item[2] for item in text_data_line])
                                    except TypeError:
                                        print('Error ', text_data_line, ' ----- ')
                                        print('Line data ', line_data)

                                    word_list.extend(text_.split(' '))
                                    text_x = min([item[3] for item in text_data_line])
                                    text_y = min([item[4] for item in text_data_line])
                                    text_xw = max([item[5] for item in text_data_line])
                                    text_yh = max([item[6] for item in text_data_line])  
                                    text_w =  text_xw - text_x
                                    text_h =   text_yh - text_y
                                    text_avg_h = round(np.average([item[8] for item in text_data_line]))
                                    string_line =  [text_, text_x, text_y, text_xw, text_yh, text_w, text_h, text_avg_h] 
                                    string_dic =  {'file_name':file_name,'page_number':i_page+page_start+1,'line_number':line,'text':text_, 'x':text_x,'y': text_y, 'x2':text_xw, 'y2':text_yh, 'w':text_w, 'h':text_h, 'avg_h': text_avg_h} 
                                    
                                    #text_lines.append(string_line)
                                    text_lines.append(string_dic)
                                    
                                    if len(line_data) == len(text_data):
                                        break

                                    index_data_line = [item[0] for item in line_data]
                                    for index_val in index_data_line:
                                        word_in_text_indexes[index_val] = 1
                                else:
                                    break
                            
                elif split_type == 'word': 
                    for data_word in data: 
                        word_list.extend(data_word[2])
                        text_line =  [data_word[2], data_word[3], data_word[4], data_word[5], data_word[6], data_word[5]-data_word[3], data_word[6]-data_word[4], data_word[8], width, height]
                        text_lines.append(text_line)
                else:
                    raise Exception('split_type =' + split_type + ' not implemented.')

                line_numbers = {item:i for i,item in enumerate(np.sort([item['y'] for item in text_lines]))}
                for i,item in enumerate(text_lines):
                    text_lines[i]['line_number'] = line_numbers[item['y']]
                text_lines.sort(key=lambda x:x['line_number'], reverse=False)
            
                # normalize the bounding boxes
                text_boxes = []
                for i,line in enumerate (text_lines):
                    normalized_bbox = normalize_box([line['x'], line['y'], line['x2'], line['y2']], width, height)
                    bbox_dic =  {'file_name':file_name,'page_number':i_page+page_start+1,'line_number':line,'n_y':normalized_bbox[0], 'n_y':normalized_bbox[1],'n_x2': normalized_bbox[2], 'n_y2':normalized_bbox[3], 'page_width':width, 'page_height':height} 
                                    
                    text_boxes.append(bbox_dic)

                page_fitz = doc_fitz[i_page+page_start]
                page_html = page_fitz.get_text("html")
                html_content = BeautifulSoup(page_html, "html.parser")

                #check if the page is empty, scanned or encoded
                content_type, read_confidence, number_images  = getHTMLInformation(html_content, page_html, word_list)

                if is_encrypted:
                    encrypted = "D"
                else:
                    encrypted = "N"

                #list of information at page level
                page_dic = {'file_name':file_name, 'page_number': i_page+page_start+1, 'page_width': width,'page_height': height, 'orientation':orientation, 'angle': rotation, 
                    'language':oLanguage[0], 'lang_confidence': str(oLanguage[1]), 'content_type': content_type, 'read_confidence':read_confidence, 'encrypted': encrypted, 'number_images':number_images}
                assert len(text_lines)==len(text_boxes)
                if len(text_lines) >0: 
                    sample_response_list[pdf_file_path]['words'].extend(text_lines)
                    sample_response_list[pdf_file_path]['bbox'].extend(text_boxes) 
                sample_response_list[pdf_file_path]['pages'].append(page_dic) 
        
        doc_fitz.close()

    return sample_response_list


def normalize_box(box, width, height):
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]


def getPageTextRotating(image, image_name, detect_lang=True, lang=None, lang_modules=None): 
    module_stopwords = lang_modules['STOP_WORDS']
    module_words =  lang_modules['WORDS']
    unified_stop_words = {}
    if lang is not None:
        unified_stop_words = {item:0 for item in module_stopwords[lang]}
    else:
        for lang_ in module_stopwords:
            unified_stop_words.update({item:0 for item in module_stopwords[lang_]})
    
    directory_temp = 'temp'
    image_name = os.path.join(directory_temp,'temp_' + image_name + '_' + str(random.randint(0,1000))  + '.jpg')
    if not os.path.exists(directory_temp):
        os.makedirs(directory_temp)
    image.save(image_name)
    empty_page = False

    try:
        osd_resp = pytesseract.image_to_osd(image_name, config ='--psm 0 -c min_characters_to_try=5 ' + '-l ' + '+'.join(list(langs.keys())), output_type=Output.DICT)
        orientation = osd_resp['orientation']
    except:
        empty_page = True
        orientation = 0
        print('possible blank page ' + image_name)
        
    if os.path.exists(image_name):
        os.remove(image_name)
    
    status = 'OK'
    if orientation !=0:
        image_temp = image.rotate(orientation, expand=True)
        status = 'Rotated'
    else:
        image_temp = image
    
    text_ = pytesseract.image_to_string(image_temp, config = '-l ' + '+'.join(list(langs.keys())) + ' --oem 3 --psm 6') 
    text_ = text_.lower()
    words = [w for line in text_.split('\n') for w in line.split(' ') if len(w.strip())>0]

    if len(words) == 0:
        return {'image': image_temp, 'rotation': 0, 'status':'Empty', 'lang': ['',0]}
    
    if len(words) >0:
        if detect_lang:
            lang = getLanguageByWords(words, module_stopwords)
            if lang[1]<0.4:
                lang = getLanguageByWords(words, module_words)
                if lang[1]==0:
                    lang = getLanguageByWords(words, module_words)
        else:
            lang = [lang, 2]


    return {'image': image_temp, 'rotation': orientation, 'status':status, 'lang': lang}
    
        

def download_lang_dictionaries():    
    for lang in PRETRAINED_VOCAB_FILES_MAP.keys():
        file_url = PRETRAINED_VOCAB_FILES_MAP[lang]
        file_name = file_url.split('/')[-1] 
        directory_location = 'dicts/'
        if not os.path.exists(directory_location):
            os.makedirs(directory_location)

        file_location = os.path.join(directory_location, lang + '_' + file_name)
        if not os.path.exists(file_location):
            response = requests.get(file_url)
            if Path(file_url).suffix == '.json':
                with open(file_location, 'w', encoding='utf-8') as oFile:
                    json.dump(json.loads(response.content.decode('utf-8')), oFile)
            else:
                with open(file_location, "w", encoding='utf-8') as oFile:
                    oFile.write(response.content.decode('utf-8'))

def get_wordsmodules():
    directory_location = 'dicts/'
    langModules_words = {}
    for lang in PRETRAINED_VOCAB_FILES_MAP.keys():
        file_url = PRETRAINED_VOCAB_FILES_MAP[lang]
        file_name = file_url.split('/')[-1] 
        file_location = os.path.join(directory_location, lang + '_' + file_name)
        
        if Path(file_location).suffix == '.json':
            with open(file_location, "r") as oFile:
                dict_words = json.load(oFile)
                dict_words = {item.split('<')[0] for item in dict_words}  
                langModules_words[lang] = {item for item in dict_words if len(item.strip())>0}  
        else:
            with open(file_location, "r", encoding='utf-8') as oFile:
                dict_words = oFile.readlines()
                dict_words = {word.split('\n')[0]:0 for word in dict_words}
                langModules_words[lang] = {item for item in dict_words if len(item.strip())>0}
 
    return langModules_words

def get_stopwordsmodules(lang=None):
    if lang is not None: 
        lang_values = list(langs.keys())[0]
    lang_values = list(langs.keys())
    langModules = {language: importlib.import_module("spacy.lang." + langs[language] + ".stop_words") for language in lang_values}
    langModules_words = {language:langModules[language].STOP_WORDS for language in langModules}
    return langModules_words

def getLanguageByWords(tokenized_text, words_modules): 
    langs_eq = list(langs.keys())
    langDicts = {}
    ratios_stopwords = []

    total_words = len(tokenized_text)
    confidence = 0 
    for lang in list(langs.keys()): 
        langDicts[lang] = set(words_modules[lang])
        for lang2 in list(langs.keys()):
            if lang2 != lang:
                langDicts[lang] -= set(words_modules[lang2])

    if total_words >0:
        for lang in list(langs.keys()):
            ratios_stopwords.append(len([word for word in tokenized_text if word in langDicts[lang]])/total_words)

        if np.sum(ratios_stopwords) >0:
            confidence = int(np.max(ratios_stopwords) / np.sum(ratios_stopwords)*100)/100
        if confidence ==0: #check how to improve this, create a dictionary of words that falls in this category and then organize per language.
            return [langs_eq[0], confidence]
        return [langs_eq[np.argmax(ratios_stopwords)], confidence]
    else: #Blank page, returning default language and confidence = 0
        return [langs_eq[0],  0]

def cleanSentences(list_sentences): 
    for i, sentence in enumerate(list_sentences):
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("‘","'")
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("—","-")
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("’","'")
        if sentence.get('text_left') is not None:
            list_sentences[i]['text_left'] = list_sentences[i]['text'] 
    return list_sentences

def getHTMLInformation(html_content, html_pageText, ocr_word_list):
    confidence,number_images = 0, 0
    
    number_words_ocr = len(ocr_word_list)
    
    list_words_html = [word.replace(u'\xa0', u' ') for line in html_content.text.split('\n') if len(line.split())>0 for word in line.split(' ') if len(word.strip())>0]
    list_words_html = [item for line in list_words_html for item in line.split(' ')]

    #count number of images in the document
    imageList = []
    if len(html_content.find_all("p"))>0: 
        for i_line, line in enumerate(html_content.find_all("img")):
            if 'src=' in str(line):
                number_images += 1  
                imageList.append(i_line) 
    #if no text was read before, is Empty
    if number_words_ocr == 0:
        confidence = 1
        if len(list_words_html) > 0:
            confidence = 0.5
        return "E", confidence, number_images #Empty, if 0.5 means that there are words, not completely empty

    ratio_words_html_ocr = len(list_words_html)/number_words_ocr
    ratio_words_ocr_html = 0

    if ratio_words_html_ocr >0:
        ratio_words_ocr_html = 1/ratio_words_html_ocr

    #check if there is encoded text
    if '&#xfffd;&#xfffd;&#xfffd' in html_pageText:
        no_encoded_text = html_pageText.replace('&#xfffd','')
        no_encoded_text = no_encoded_text.replace(';',' ')
        no_encoded_html_content = BeautifulSoup(no_encoded_text, "html.parser")
        number_words_html = sum([len(line.split(' ')) for line in no_encoded_html_content.text.split('\n') if len(line.strip())>0])
        if number_words_html > number_words_ocr:
            number_chars_html = sum([len(line.replace(' ','')) for line in no_encoded_html_content.text.split('\n') if len(line.strip())>0])
            number_chars_words = sum([len(word) for word in ocr_word_list])
            confidence = int((number_chars_words-number_chars_html)/number_chars_words*100)/100 
            if np.floor(confidence*100) == 0:
                confidence = int(len(html_pageText.split('&#xfffd'))/number_chars_words*10000)/10000
        else:
            confidence = int((number_words_ocr-number_words_html)/number_words_ocr*100)/100 #some quantities are less than zero
        return 'X', confidence, number_images #Encoded, confidence is how much of the document is encoded
  
    if ratio_words_html_ocr < 0.5:
        confidence = int((number_words_ocr - len(list_words_html))/number_words_ocr*100)/100
        return 'S', confidence, number_images  #Scanned, confidence of how much of the document is scanned
    else:
        confidence = int(min(ratio_words_html_ocr,ratio_words_ocr_html)*100)/100 
        if ratio_words_html_ocr < 0.6 or ratio_words_ocr_html < 0.6:
            return 'C', confidence, number_images #Bad reading, how much of the document could have a bad reading
    
        if ratio_words_html_ocr < 0.75 or ratio_words_ocr_html < 0.75:
            return 'B', confidence, number_images #Possible Bad Reading 
    
        return 'R', min(confidence,1), number_images #Readable PDF

    
def decryptPDF(fileName):
    oPathFile =  Path(fileName)
    destinationDirectory = os.path.join(oPathFile.parent, "decrypted")
    destination = os.path.join(destinationDirectory, fileName)

    if os.path.exists(destination):
        return destination

    if not os.path.exists(destinationDirectory):
        os.mkdir(destinationDirectory)
    
    try:
        with pikepdf.open(fileName) as pdf:
            pdf.save(destination)
            return destination
    except:
        return None

    
class FDExTGenerator():
    
    def __init__(self, data_dir, output_dir, split_type, filetype):
        self.output_dir = output_dir  
        self.split_type = split_type
        self.data_dir = data_dir
        self.createFileNames(filetype)
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        self.loaded_docs = {}

        if os.path.exists(self.pageInfo_filename):
            if filetype =='json':
                 with open(self.pageInfo_filename, ) as f:
                    loaded_pages = json.load(f)
                    self.loaded_docs = {doc['file_name']:0 for doc in loaded_pages} 

            elif filetype == 'txt':
                with open(self.pageInfo_filename, 'r') as f:
                    loaded_pages =  f.readlines()
                    if len(loaded_pages) > 0: 
                        loaded_pages = [page.split('\t')[:2] for page in loaded_pages] 
                        self.loaded_docs = {doc[0]:0 for doc in loaded_pages}


    def createFileNames(self, filetype="json"):
        self.text_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '.' + filetype)  
        self.bbox_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '_bbox.' + filetype) 
        self.format_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '_format.' + filetype)
        self.pageInfo_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '_page_info.' + filetype) 
        self.docInfo_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '_doc_info.' + filetype)
        self.ttype_filename = os.path.join(self.output_dir, 'text_' + self.split_type + '_ttype.' + filetype)
    
    
    def getListFiles(self, content_list):
        return [oFile for oFile in content_list if oFile.suffix in [".pdf",".png",".jpg","jpeg"] and self.loaded_docs.get(oFile.stem) is None]

    def exploreDirectory(self, data_dir):
        content_list = [Path(file_name) for file_name in os.listdir(data_dir)]
        #check if the main directory contains directories
        directory_list = [oFile for oFile in os.listdir(self.data_dir) if not os.path.isfile(oFile)]
        if len(directory_list) ==0:
            file_list = self.getListFiles(content_list)
        else:
            file_list = []
            for root, subdirs, files_ in os.walk(data_dir):
                content_list = [Path(os.path.join(root,file_name)) for file_name in files_]
                files = self.getListFiles(content_list)
                if len(files)>0:
                    file_list.extend(files)
        return file_list


    def appendToList(self, dataset_list, response_list):
        dataset_list['words'].extend(response_list['words'])
        dataset_list['bbox'].extend(response_list['bbox']) 
        dataset_list['pages'].extend(response_list['pages']) 
        return dataset_list

    def distribute(self, distrib_tool, dataset_batch, oPool=None):
        if distrib_tool =='ray':
            results = oPool.map(apply_ocr_and_html, [l_batch for l_batch in dataset_batch])
        else:
            oPool = mp.Pool(self.total_workers, maxtasksperchild=1)
            results = oPool.map(apply_ocr_and_html, dataset_batch)
            oPool.close()

        return results

    def distribute_batch(self, oPool, distrib_tool, dataset_batch, max_pages_per_run):
        print('Distribute work of ' + str(len(dataset_batch)) + ' batches.')
                    
        start_time = time.time()
        dataset_list = {'words':[], 'bbox':[], 'pages':[]}
        
        results = self.distribute(distrib_tool, dataset_batch, oPool)

        end_time = time.time() - start_time
        
        print('Collecting results from ' + str(len(dataset_batch)) + ' batches. Total processing time: ' + str(end_time/60) + ' minutes.')
        
        for i_result, document_list in enumerate(results): 
            for i_doc, doc_result in enumerate(document_list):
                if len(document_list[doc_result]['pages']) >0:
                    last_page = document_list[doc_result]['last_page'] 
                    total_pages = document_list[doc_result]['total_pages'] 

                    dataset_list = self.appendToList(dataset_list,document_list[doc_result])

                    if last_page < total_pages-1: 
                        number_groups = int(np.ceil((total_pages - last_page)/(self.total_workers*max_pages_per_run)))
                        print('Parallelizing large document with ', total_pages, ' pages (', last_page, ' already processed) ' , number_groups ,' groups')
                        start_page_worker = last_page
                        last_page_worker = last_page
                        for group_n in range(number_groups):
                            dataset_batch_list_new = []
                            for worker in range(self.total_workers):
                                last_page_worker += max_pages_per_run
                                document_batch_new = dataset_batch[i_result]['data'][i_doc].copy()
                                document_batch_new['page_start'] = start_page_worker
                                document_batch_new['page_end'] = last_page_worker
                                dataset_batch_list_new.append({'data':[document_batch_new], 'modules':dataset_batch[i_result]['modules']})
                                start_page_worker += max_pages_per_run
                                
                            results_w = self.distribute(distrib_tool, dataset_batch_list_new, oPool)
                            print('Receiving distributed results of group ', group_n, "/", number_groups)
                            for result_w in results_w:
                                for doc_result_temp in result_w:
                                    dataset_list['words'].extend(result_w[doc_result_temp]['words'])
                                    dataset_list['bbox'].extend(result_w[doc_result_temp]['bbox']) 
                                    dataset_list['pages'].extend(result_w[doc_result_temp]['pages']) 
                    
                            assert len(dataset_list['words']) == len(dataset_list['bbox']) 
                    
                    assert len(dataset_list['words']) == len(dataset_list['bbox']) 
                                       
                    #save files
                    save_data(dataset_list['pages'], self.pageInfo_filename)
                    if len(dataset_list['words'])>0:
                        save_data(dataset_list['words'], self.text_filename)
                        save_data(dataset_list['bbox'], self.bbox_filename)
                        dataset_list = {'words':[], 'bbox':[], 'pages':[]}

                else:
                    print('ERROR No readable results. ' , list(document_list.keys()))

    def generateFiles(self, detect_lang, lang, worker_load, total_workers,max_docs_per_run, max_pages_per_run,distrib_tool): 
        if distrib_tool =='ray':
            oPool = Pool()
            print("Nodes in the Ray cluster:", ray.nodes())
            if os.environ.get('ip_head') is not None:
                ray.init(address=os.environ["ip_head"], num_cpus = total_workers*ray.nodes(), ignore_reinit_error=True)
            else:
                ray.init(num_cpus = total_workers, ignore_reinit_error=True)
        else:
            oPool = None

        download_lang_dictionaries()
        modules = {}
        modules['WORDS'] = get_wordsmodules()
        modules['STOP_WORDS'] = get_stopwordsmodules()
        
        self.detect_lang = detect_lang
        self.lang = lang
        self.worker_load = worker_load
        self.total_workers = total_workers

        file_list = self.exploreDirectory(self.data_dir)

        dataset_list = {'words':[], 'bbox':[], 'pages':[]}
        print('Already loaded ',  len(self.loaded_docs), 'documents in the dataset')
        print('Starting generation process for ' + str(len(file_list)) + ' documents')
        dataset_batch = []
        dataset_worker_batch = []
        for i_file in tqdm.tqdm(range(min(max_docs_per_run, len(file_list)))):
            oFile = file_list[i_file] 
            print('Append document ', oFile.name , i_file ,' to list.')
            dataset_worker_batch.append({'datafile':str(oFile), 'split_type':self.split_type, 'detect_lang': self.detect_lang, 'lang':self.lang,'page_start':0,'page_end': max_pages_per_run})
            if len(dataset_worker_batch) % self.worker_load ==0 or len(dataset_worker_batch) == len(file_list):
                dataset_batch.append({'data': dataset_worker_batch, 'modules': modules})
                dataset_worker_batch = []
                print('Append list of ', len(dataset_batch[0]), 'documents to worker ', len(dataset_batch))
                if len(dataset_batch) % self.total_workers==0 or len(dataset_batch) == len(file_list):
                    self.distribute_batch(oPool, distrib_tool, dataset_batch, max_pages_per_run)



def save_data(data_list, file_name, filetype='json'):
    append_header = False
    if not os.path.exists(file_name):
        append_header = True
    if filetype =='json': 
        existing_data = []
        if not append_header:        
            with open(file_name, 'r') as file:    
                existing_data = json.load(file) 
        
        existing_data.extend(data_list)

        with open(file_name, 'w') as file:    
            json.dump(existing_data, file)
            print('updating file ', file_name)
    elif filetype =='txt':
        with open(file_name, 'a', encoding='utf-8') as file:   
            if append_header:     
                header = '\t'.join([item for item in row])
                file.write(header + '\n')
            for row in data_list:
                row_content = '\t'.join([str(row[item]) for item in row]) + '\n'
                file.write(row_content)
            print('updating file ', file_name)
class FDExt(Dataset):

    def __init__(self, data_dir, output_dir): 
        self.text = []
        self.all_input_ids = []
        self.all_input_mask = []
        self.all_segment_ids = []
        self.all_label_ids = []
        self.labels2idx = {}
        self.weights = {}
        self.weight_strategy = None

        self.output_dir = output_dir
        self.data_dir = data_dir

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
 

    def loadDataset(self, tasks_= 'default', filter_last_doc=None, filter_type_doc=None, filter_industry_cia=None , filter_lang=None, only_pageLevel = False):
        '''
        tasks_: Data to return togueter with the text
        doc_filter: If there is a document metadata file, a set of filters can be specified in a dictionary that could have the following keys:
        - last_doc: Only the last doc of the same company id is returned, can have any value.
        - type_doc: Only documents with the specified value (document type)
        - cia_ind: Only documents that belongs to the specified value (industry)
        '''
        

        doc_filters = {}

        if filter_last_doc:
            doc_filters['last_doc'] = True
        if filter_type_doc is not None: 
            doc_filters['type_doc'] =  filter_type_doc
        if filter_industry_cia is not None: 
            doc_filters['ind_cia'] =  filter_industry_cia
        if filter_lang is not None: 
            doc_filters['lang'] =   filter_lang

        if not os.path.exists(self.data_dir):
            raise Exception('The data directory does not exists. ')
        else:
            self.get_datarecords(self.data_dir, tasks_=tasks_, doc_filters=doc_filters, only_pageLevel=only_pageLevel) #other tasks: bbox, images+bbox, format+bbox
           
            

    def get_datarecords(self, data_dir, labels_path = None, tasks_ = 'default', doc_filters=None, only_pageLevel = False): 
        if labels_path is not None and not os.path.exists(labels_path): raise Exception("Labels' file doesn't exists")
        data = {}
        for file in os.listdir(data_dir):
                oFile = Path(file)
                if 'text_' == oFile.name[:5] and  oFile.suffix == '.txt':
                    if '_bbox'== oFile.stem[-5:] and not only_pageLevel:
                        if 'bbox' in tasks_ :
                            with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as bbox_file:
                                data['bbox'] = bbox_file.readlines() 
                    elif '_format'== oFile.stem[-7:] and not only_pageLevel:
                        if 'format' in tasks_:  
                            with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as format_file:
                                data['format'] = format_file.readlines()
                    elif '_page_info'== oFile.stem[-10:]: 
                        with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as page_info_file:
                            data['page_info'] = page_info_file.readlines()
                    elif '_doc'== oFile.stem[-4:] and not only_pageLevel: 
                        with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as doc_info_file:
                            data['doc_info'] = doc_info_file.readlines()
                    elif '_cia'== oFile.stem[-4:] and not only_pageLevel:
                        with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as cia_info_file:
                            data['cia_info'] = cia_info_file.readlines()
                    elif not only_pageLevel and len(oFile.stem.split("_"))==2: 
                        with open(os.path.join(data_dir, oFile.name), encoding='utf-8') as text_file:
                            data['text'] = text_file.readlines()
                        

        if data.get('text') is None and not only_pageLevel: raise Exception('Text file not found.')
        if data.get('bbox') is not None:
            assert len(data['text']) == len(data['bbox'])
        if data.get('format') is not None:
            assert len(data['text']) ==len(data['format'])
        
        if labels_path is not None:
            with open(labels_path) as labels_file:
                data['labels'] = labels_file.readlines()
        
        if not only_pageLevel:
            dataset = self.create_dict_from_datatext(data['text'], types_list=[str, int, int, str, int, int, int, int, float, float, float]) 
            self.text = [item['text'] for item in dataset]
        else:
            dataset = {}

        if data.get('bbox') is not None:
            print('working with bbox')
            dataset_tmp = self.create_dict_from_datatext(data['bbox'], types_list=[str, int, int, str, float, float, float]) 
            dataset = self.merge_dataset(dataset, dataset_tmp, level='record')
            data['bbox'] = None
            del dataset_tmp

        if data.get('format') is not None:
            print('working with format')
            dataset_tmp = self.create_dict_from_datatext(data['format'], types_list=[str, int, int, str]) 
            data['format'] = None
            dataset = self.merge_dataset(dataset, dataset_tmp, level='record')
            del dataset_tmp

        if data.get('page_info') is not None:
            print('working with page info')
            dataset_pages = self.create_dict_from_datatext(data['page_info'], types_list=[str, int, float, float, str, int, str, float, str, float, str, int]) 
            data['page_info'] = None
            dataset = self.merge_dataset(dataset, dataset_pages, level='page')
            self.page_info = dataset_pages
            del dataset_pages

        if data.get('doc_info') is not None:
            print('working with doc info')
            dataset_docs = self.create_dict_from_datatext(data['doc_info'], types_list=[str, int, int])  
            dataset = self.merge_dataset(dataset, dataset_docs, level='doc')
            data['doc_info'] = dataset
            if self.page_info is not None:
                self.page_info =  self.merge_dataset(self.page_info , dataset_docs, level='doc')
            else:
                self.page_info = dataset_docs
            del dataset_docs

        if data.get('cia_info') is not None:
            dataset_docs = self.create_dict_from_datatext(data['cia_info'], types_list=[str])
            data['cia_info'] = None
            dataset = self.merge_dataset(dataset, dataset_docs, level='cia')
            del dataset_docs

        if doc_filters is not None and (data.get('doc_info') is not None or data.get('page_info') is not None):
            self.dataset = dataset
            dataset = self.filter_dataset(doc_filters) 
        self.dataset = dataset
 

    def filter_dataset(self, doc_filters):
        dataset_final_t0 = []
        if doc_filters.get('lang') is not None:
            for document in self.dataset:
                if doc_filters.get('lang') == document['language']:
                    dataset_final_t0.append(document) 
        else:
            dataset_final_t0 = self.dataset

        dataset_final_t1 = []
        if doc_filters.get('type_doc') is not None:
            for document in dataset_final_t0:
                if document.get('document_type') is not None:
                    if doc_filters.get('type_doc') in document['document_type']:
                        dataset_final_t1.append(document)
            del dataset_final_t0
        else:
            dataset_final_t1 = dataset_final_t0

        dataset_final_t2 = []
        if doc_filters.get('ind_cia') is not None:
            for document in dataset_final_t1:
                if document.get('company_industry') is not None:
                    if doc_filters.get('ind_cia') == document['company_industry']:
                        dataset_final_t2.append(document)
            del dataset_final_t1
        else:
            dataset_final_t2 = dataset_final_t1
            
        dataset_final_t3 = []
        if doc_filters.get('last_doc'):
            company_dic = {}
            for document in dataset_final_t2: 
                if document.get('company_id') is not None:
                    company_id = document['company_id']
                    company_dic[company_id] =  document['document_year'] if company_dic.get(company_id) is None else max(document['document_year'], company_dic[company_id])
            for document in dataset_final_t1:
                if document.get('document_year') is not None: 
                    if document['document_year'] == company_dic[document['company_id']]:
                        dataset_final_t3.append(document)
        else:
            dataset_final_t3 = dataset_final_t2
        
        self.dataset = dataset_final_t3 
        return dataset_final_t3

    def create_dict_from_datatext(self, data, types_list=None):
        dataset = []
        headers= {i:item.replace(' ','_').lower() for i,item in enumerate(data[0][:-1].split('\t'))}
        if len(types_list) < len(headers): types_list = types_list + [str for item in range(len(headers)-len(types_list))]
        ini_len_header = len(headers)
        multiheader_index =[item for  item in headers if len(headers[item].split(','))>1]
        add_new_headers = False
        if len(multiheader_index) >0:
            add_new_headers = True
            attached_headers =  {i+ini_len_header:header for i, header in enumerate(headers[multiheader_index[0]].split(','))}
            headers.update(attached_headers)
        print('Total rows ', len(data))
        for row in tqdm.tqdm(range(1, len(data))): 
            data_vec = data[row][:-1].split('\t')
            info = {headers[i]:types_list[i](item) for i,item in enumerate(data_vec)}
            if add_new_headers:
                new_data = {headers[i+ini_len_header]:types_list[-1](item) for i, item in enumerate(info[headers[multiheader_index[0]]].split(','))}
                info.update(new_data)
                info.pop(headers[multiheader_index[0]])
            dataset.append(info)
        return dataset

    def merge_dataset(self, dataset, dataset_tmp, level='record'):
        if level == 'record':
            for i_record, record in enumerate(dataset):
                dataset[i_record].update(dataset_tmp[i_record])
        elif level=='page':
            pages_not_found = {}
            dataset_tmp_dict = {item['file_name'] + '_' + str(item['page_number']):item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                page_id = record['file_name'] + '_' + str(record['page_number'])
                if dataset_tmp_dict.get(page_id) is not None:
                    dataset[i_record].update(dataset_tmp_dict[page_id])
                else:
                    pages_not_found[record['file_name'] + "_" + str(record['page_number'])] = 0 
            del dataset_tmp_dict
    
            if len(pages_not_found) >0:
                print('Documents not found: ', list(pages_not_found.keys()))
    
        elif level=='doc':
            docs_not_found = {}
            dataset_tmp_dict = {item['file_name']:item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                if dataset_tmp_dict.get(record['file_name']) is not None:
                    dataset[i_record].update(dataset_tmp_dict[record['file_name']])
                else:
                    docs_not_found[record['file_name']] = 0
            del dataset_tmp_dict
    
            if len(docs_not_found)>0:
                print('Documents not found: ', list(docs_not_found.keys()))
    
        elif level=='cia':
            companies_not_found = {}
            documents_without_companies= {}
            dataset_tmp_dict = {item['company_id']:item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                if record.get('company_id') is not None:
                    if dataset_tmp_dict.get(record['company_id']) is not None:
                            dataset[i_record].update(dataset_tmp_dict[record['company_id']])
                    else:
                        companies_not_found[record['company_id']] = 0 
                else:
                    documents_without_companies[record['file_name']] = 0
            del dataset_tmp_dict

            if len(companies_not_found)>0:
                print('Companies not found: ', list(companies_not_found.keys()))
                print('Documents without companies: ', list(documents_without_companies.keys()))
    
        return dataset

    def getTextListByDocument(self, dataset_perc=1, group_by_page= False):
        doc_list = {item['file_name']:0 for item in self.dataset}
        dataset_size = int(len(doc_list) * min(1,dataset_perc))
        documents_detail= {}
        company_number = 0
        row_number =0
        while len(documents_detail) < dataset_size and company_number < dataset_size:
            current_row = self.dataset[row_number]
            file_name = current_row['file_name']
            page_number = current_row['page_number']
            if documents_detail.get(file_name) is None:                 
                documents_detail[file_name] = {} if group_by_page else []
                company_number +=1 
            if group_by_page:
                if documents_detail[file_name].get(page_number) is None:
                    documents_detail[file_name][page_number] = []
                documents_detail[file_name][page_number].append(current_row)
            else:
                documents_detail[file_name].append(current_row)
            row_number += 1

        return documents_detail


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default=None,
        type=str,
        required=True,
        help="Action to execute. Options: generate, load ", )
    
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The data dir where the pdf files are located.", )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The data dir where the output files are going to be saved.", )

    parser.add_argument(
        "--metadata_file",
        default=None,
        type=str,
        required=False,
        help="If there is metadata information at document level, the file name should be specified.", )

    parser.add_argument(
        "--split_type",
        default="line",
        type=str,
        required=False,
        help="Level of granularity for generation of each dataset record. Line or Word.", )

    parser.add_argument(
        "--detect_lang",
        default=False,
        type=bool,
        required=False,
        help="True if the generator is going to detect the language of each page. Otherwise, lang parameter is going to be considered.", )
    
    parser.add_argument(
        "--lang",
        default='fra',
        type=str,
        required=False,
        help="To avoid language detection. Options: fra, eng, deu, swe.", )
    
    parser.add_argument(
        "--total_workers",
        default=14,
        type=int,
        required=False, 
        help="For parallel processing, if the number is bigger than the number of processors available, the maximum number of processor are going to be considered.", )

    parser.add_argument(
        "--worker_load",
        default=14,
        type=int,
        required=False, 
        help="For parallel processing, if the number is bigger than the number of processors available, the maximum number of processor are going to be considered.", )

    parser.add_argument(
        "--filter_last_doc",
        default=None,
        type=bool,
        required=False, 
        help="In Loading, only for considering the last document of the company.", )

    parser.add_argument(
        "--filter_type_doc",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document of the selected type.", )
    
    parser.add_argument(
        "--filter_industry_cia",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document's company of the selected industry.", )
    
    parser.add_argument(
        "--filter_lang",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document in the specified language.", )

    
    parser.add_argument(
        "--max_pages",
        default=200,
        type=int,
        required=False, 
        help="Number of maximum number of files to be processed.", )

    parser.add_argument(
        "--max_docs",
        default=10000,
        type=int,
        required=False, 
        help="Number of maximum number of files to be processed.", )

    parser.add_argument(
        "--distrib_tool",
        default="ray",
        type=str,
        required=False, 
        help="The tool for redistributing the parallel work. Ray or pool.", )

        
    parser.add_argument(
        "--filetype",
        default="json",
        type=str,
        required=False, 
        help="Extension for reading/generating the files (json, txt).", )


    args = parser.parse_args()

    
    print('Cuda available ',torch.cuda.is_available())
    print('Number of available GPUS ', torch.cuda.device_count())
    print('Number of processors ' + str(mp.cpu_count()))

    if args.action == 'generate':
        oFDExTGenerator = FDExTGenerator(args.data_dir, args.output_dir, args.split_type, args.filetype)
        oFDExTGenerator.generateFiles(args.detect_lang, args.lang, args.worker_load, args.total_workers, args.max_docs, args.max_pages, args.distrib_tool) 
    elif args.action =='load':
        oFDExt = FDExt(args.data_dir, args.output_dir)
        oFDExt.loadDataset(filter_last_doc=args.filter_last_doc, filter_type_doc=args.filter_type_doc, filter_industry_cia=args.filter_industry_cia , filter_lang=args.filter_lang)
        print()