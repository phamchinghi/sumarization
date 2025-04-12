import os
import nltk
from bs4 import BeautifulSoup
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import svm
nltk.download('averaged_perceptron_tagger')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# đọc file từ path
def read_files(path):
    file_data = open(path, "r")
    data = file_data.readlines()
    return data
# Đọc single file
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

# convert string to tag để lấy các value docid, num, wdcount
def convert_tag(tag_value):
    soup = BeautifulSoup(tag_value, "html.parser")
    return soup.findAll('s')


# remove html markup
def remove_html_markup(html):
    str_bgn = '">'
    bgn = html.index(str_bgn)
    end = html.index('</s>')
    return html[bgn + len(str_bgn):end]


# Đặc trưng 2 - Fixed-Phrase Feature indicator: check các từ chỉ định
def is_indicator(text):
    indicators = ['finally', 'in a word', 'in brief', 'briefly', 'in conclusion', 'in the end', \
                  'in the final analysis', 'on the whole', 'thus', 'to conclude', 'to summarize', \
                  'in sum', 'to sum up', 'in summary', 'to reiterate', 'this letter', 'to summarise', \
                  'to end', 'to close', 'last of all', 'last but not least', 'at in all' , 'in short', \
                  'in brief', 'briefly', 'to bring about', 'to bring about', 'for this reason', 'all these reasons']
    indicator = False
    for word in indicators:
        if word in text.lower():
            indicator = True
            break
    return indicator


# Đặc trưng 1 - Sentence Length Cut-off: Feature true if sentence > 5 words
def is_length_valid(text):
    items = convert_tag(text)
    for item in items:
        if int(item["wdcount"]) > 5:
            return True
        else:
            return False


# Đặc trưng 3 - tìm vị trí của đoạn: 0: initial, 1: medial, 2: final
def get_paragraph_feature(text, dictionary):
    items = convert_tag(text)
    paragraph_pos = 1
    for item in items:
        lst = dictionary.get(item["docid"])
        value_num = int(item["num"])
        if value_num == lst[0]:
            paragraph_pos = 0
        elif value_num == lst[1]:
            paragraph_pos = 2

    return paragraph_pos


# tìm min value trong list có cùng docid
def get_min_value(min_value, item):
    if min_value > item:
        min_value = item
    return min_value


# tìm max value trong list có cùng docid
def get_max_value(max_value, item):
    if max_value < item:
        max_value = item
    return max_value


# tạo 1 list chỉ chứa các tag <s>
def filter_list_tag(text_files):
    filter_files = []

    for text in text_files:
        if "P>" in text:
            continue
        elif "TEXT>" in text:
            continue
        elif "SUBJECT>" in text:
            continue
        elif "TYPE>" in text:
            continue
        elif "GRAPHIC>" in text:
            continue
        elif "XX>" in text:
            continue
        elif "CO>" in text:
            continue
        elif "CN>" in text:
            continue
        elif "IN>" in text:
            continue
        elif "PUB>" in text:
            continue
        elif "PAGE>" in text:
            continue
        elif "HEAD>" in text:
            continue
        elif "BYLINE>" in text:
            continue
        elif "COUNTRY>" in text:
            continue
        elif "CITY>" in text:
            continue
        elif "EDITION>" in text:
            continue
        elif "CODE>" in text:
            continue
        elif "NAME>" in text:
            continue
        elif "PUBDATE>" in text:
            continue
        elif "DAY>" in text:
            continue
        elif "MONTH>" in text:
            continue
        elif "PG.COL>" in text:
            continue
        elif "PUBYEAR>" in text:
            continue
        elif "REGION>" in text:
            continue
        elif "FEATURE>" in text:
            continue
        elif "STATE>" in text:
            continue
        elif "WORD.CT>" in text:
            continue
        elif "DATELINE>" in text:
            continue
        elif "COPYRGHT>" in text:
            continue
        elif "LIMLEN>" in text:
            continue
        elif "LANGUAGE>" in text:
            continue
        elif "NOTE>" in text:
            continue
        elif "TABLE>" in text:
            continue
        elif "ROWRULE>" in text:
            continue
        elif "TABLEROW>" in text:
            continue
        elif "CELLRULE>" in text:
            continue
        elif "TABLECELL>" in text:
            continue
        elif "F>" in text:
            continue
        filter_files.append(text)

    return filter_files

# Tách file thành các đoạn theo cùng docid cho việc kiểm tra vị trí câu
def filter_list_docid(text_files):
    list_tag = {}
    list_tf_idf = {}
    list_num = {}

    for text in text_files:
        items = convert_tag(text)
        for item in items:
            key = item["docid"]
            if key not in list_tag:
                list_tag[key] = []
                list_tf_idf[key] = []
                list_num[key] = []

            list_tag[key].append(item)

            # list tfidf
            item_convert = get_tf_idf_value(item)
            list_tf_idf[key].append(item_convert)

            # list chứa tất cả num trong cùng docid
            list_num[key].append(int(item["num"]))

    return list_tag, list_tf_idf, list_num


# lấy câu hoàn chỉnh để tính tfidf
def get_tf_idf_value(item):
    item_str = remove_html_markup(str(item))
    last_index = 0
    if item_str.endswith("."):
        last_index = item_str.index(".")
    elif item_str.endswith("'"):
        last_index = item_str.index("'")
    item_remove = item_str[0:last_index]
    item_convert = ""
    item_splits = item_remove.split(" ")
    for item_split in item_splits:
        if item_split.endswith(","):
            item_split = item_split[0:item_split.index(",")]
        elif item_split.endswith("''"):
            item_split = item_split[0:item_split.index("''")]
        item_convert += item_split + " "

    return item_convert


# tìm min max trong docid
def get_list_min_max(list_num):
    # tìm min max trong docid
    list_min_max = {}
    for key_num, value_nums in list_num.items():
        list_min_max[key_num] = []
        is_intial = True
        for value_num in value_nums:
            if is_intial:
                value_min = value_num
                value_max = value_num
                is_intial = not is_intial
                continue

            value_min = get_min_value(value_min, int(value_num))
            value_max = get_max_value(value_max, int(value_num))
            list_min_max[key_num] = value_min, value_max

    return list_min_max


# tìm list tf idf
def get_list_tf_idf(lst_tf_idf_org):
    word_vectorizer = TfidfVectorizer(
        stop_words='english',
        sublinear_tf=True,
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 1),
        token_pattern='[a-zA-Z0-9_,.-]\\S+',
        max_features=30000)

    tf_idf_dict = {}
    for key, value in lst_tf_idf_org.items():
        tf_idf_dict[key] = []
        X = word_vectorizer.fit_transform(value)
        idf = word_vectorizer.idf_
        tf_idf_new_dict = dict(zip(word_vectorizer.get_feature_names_out(), idf))
        sorted_tf_idf = sorted(tf_idf_new_dict.items(), key=lambda kv: kv[1])
        sorted_tf_idf = sorted_tf_idf[0:15]
        tf_idf_docs = ""
        for tf_idf in sorted_tf_idf:
            tf_idf_docs += tf_idf[0] + " "
        tf_idf_dict[key] = tf_idf_docs

    return tf_idf_dict


# Đặc trưng 5 - Uppercase word: kiểm tra câu có proper names
def is_exist_proper_name(text):
    clean_text = remove_html_markup(text)
    tagged_sent = pos_tag(clean_text.split())
    proper_nouns = [word for word, pos in tagged_sent if pos == 'NNP']
    if len(proper_nouns) > 2:
        return True
    else:
        return False


# kiểm tra file có tồn tại trong folder sum
def is_exist_file_sum(element, collection: iter):
    return element in collection


# Đặc trưng thứ 4 - kiểm tra đoạn có Thematic word
def is_thematic_word(tf_idfs, line):
    thematic = False
    items = convert_tag(line)
    for item in items:
        lst_tf_idf = tf_idfs.get(str(item["docid"]))
        for tf_idf in lst_tf_idf:
            if tf_idf in remove_html_markup(line).lower():
                thematic = True
                break

    return thematic


# get vector của file can rút trích
def get_vector(path_text, file_name_text):
    text_files = read_files(path_text + file_name_text)
    filter_files = filter_list_tag(text_files)

    # tao list theo docid
    list_text, lst_tf_idf_org, lst_num = filter_list_docid(filter_files)

    # tim min max cho kiem tra vi tri cua doan
    lst_min_max = get_list_min_max(lst_num)

    # tf_idf tim thematic word
    lst_tf_idf = get_list_tf_idf(lst_tf_idf_org)

    trained_C = []

    for line_text in filter_files:
        feature1, feature2, feature3, feature4, feature5 = extract_vector(line_text, lst_min_max, lst_tf_idf)
        vec = (feature1, feature2, feature3, feature4, feature5)
        trained_C.append(vec)

    return trained_C


# Lấy ra vector của mỗi dòng theo đặc trưng
def extract_vector(line_text, lst_min_max, lst_tf_idf):
    feature1, feature2, feature3, feature4, feature5 = 0, 0, 0, 0, 0

    #Đặc trưng 1
    if is_indicator(line_text):
        feature1 = 1

    # Đặc trưng 2
    if is_length_valid(line_text):
        feature2 = 1

    #Đặc trưng 3
    paragraph_feature = get_paragraph_feature(line_text, lst_min_max)
    if paragraph_feature == 1:
        feature3 = 1
    elif paragraph_feature == 2:
        feature3 = 2

    #Đặc trưng 4
    if is_thematic_word(lst_tf_idf, line_text):
        feature4 = 1

    # Đặc trưng 5
    if is_exist_proper_name(line_text):
        feature5 = 1

    return feature1, feature2, feature3, feature4, feature5


if __name__ == '__main__':
    try:
        # config path and file name
        path_text = os.getcwd() + "\\DUC_TEXT\\"
        path_sum = os.getcwd() + "\\DUC_SUM\\"

        # scikit-learn
        trained_X = []
        trained_Y = []

        # Khai báo các giá trị tham số cần thử nghiệm
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Thử các giá trị C khác nhau
            'kernel': ['linear', 'rbf', 'poly']  # Thử các loại hạt nhân khác nhau
        }

        directory_text = path_text
        directory_sum = path_sum
        for file_name_text in os.listdir(directory_text):
            text_files = read_files(path_text + file_name_text)

            # tạo list chỉ chứa tag <s> để tìm đặc trưng
            filter_files = filter_list_tag(text_files)

            # tạo list theo docid
            list_text, lst_tf_idf_org, lst_num = filter_list_docid(filter_files)

            # tìm min, max để kiểm tra vị trí của câu
            lst_min_max = get_list_min_max(lst_num)

            # tf_idf tìm thematic word
            lst_tf_idf = get_list_tf_idf(lst_tf_idf_org)

            for line_text in filter_files:
                feature1, feature2, feature3, feature4, feature5 = extract_vector(line_text, lst_min_max, lst_tf_idf)
                feature6 = 0

                # kiểm tra đoạn văn bản có tồn tại folder sum
                for file_name_sum in os.listdir(directory_sum):
                    if file_name_text in file_name_sum:
                        sum_files = read_files(path_sum + file_name_sum)
                        if is_exist_file_sum(line_text, sum_files):
                            feature6 = 1
                    else:
                        continue

                vec1 = (feature1, feature2, feature3, feature4, feature5)
                vec2 = feature6

                trained_X.append(vec1)
                trained_Y.append(vec2)

        # Train scikit-learn
        clf = svm.SVC(C=0.1, kernel='linear')   #Điều chỉnh thành 0.1 và linear để xem có cải thiện gì không
        # Gán trọng số để train
        weight_sum = compute_sample_weight(class_weight='balanced', y=trained_Y)
        clf.fit(trained_X, trained_Y, sample_weight=weight_sum)
        file_name_text = "d061j"
        file_content = read_files(path_text + file_name_text)
        trained_C = get_vector(path_text, file_name_text)
        test_Y = clf.predict(trained_C)

        # svm_model = svm.SVC()
        #
        # # Sử dụng GridSearchCV để thử nghiệm các giá trị tham số và tìm ra giá trị tối ưu
        # grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
        # grid_search.fit(trained_X, trained_Y)
        #
        # # In ra giá trị tham số tốt nhất
        # print("Best parameters found:", grid_search.best_params_)

        # In ra văn bản tóm tắt từ vector đặc trưng
        with open(os.getcwd() + "\\SUM\\" + file_name_text +"_sum", "w") as file:
            for i, prediction in enumerate(test_Y):
                if prediction == 1:  # Nếu dự đoán là 1 (hoặc nhãn khác tùy thuộc vào cài đặt)
                    file.write(str(file_content[i]))
        # print(test_Y)
    except ValueError as ex:
        print("Error:", ex)
