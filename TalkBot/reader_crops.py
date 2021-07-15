import linecache

def crops_reader(raw: list,path: str):
    question = []
    anwser = []
    for times in range(len(raw)):
        try:
            file = open(raw[times],'r',encoding="utf-8").read()
        except Exception as e:
            raise FileNotFoundError(e)
        if "|" in linecache.getline(raw[times],1):
            for line_times in range(len(file.split("\n"))):
                qa_list = linecache.getline(raw[times],line_times + 1).split("|")
                if len(qa_list) == 2:
                    question.append(qa_list[0])
                    anwser.append(qa_list[1])
                if len(qa_list) == 3:
                    question.append(qa_list[1])
                    anwser.append(qa_list[2])
                
        elif "#" in linecache.getline(raw[times],1):
            for line_times in range(len(file.split("\n"))):
                qa_list = linecache.getline(raw[times],line_times + 1).split("#")
                if len(qa_list) == 2:
                    question.append(qa_list[0])
                    anwser.append(qa_list[1])
                if len(qa_list) == 3:
                    question.append(qa_list[1])
                    anwser.append(qa_list[2])
    file = open(f'{path}/question','w',encoding="utf-8")
    for times in range(len(question)):
        file.write(f"{question[times]}\n")
    file.close()
    file = open(f'{path}/anwser','w',encoding="utf-8")
    for times in range(len(anwser)):
        file.write(f"{anwser[times]}\n")
    file.close()

    return "READ SUCCESSFUL"