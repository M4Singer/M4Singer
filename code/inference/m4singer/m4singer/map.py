def m4singer_pinyin2ph_func():
    pinyin2phs = {'AP': '<AP>', 'SP': '<SP>'}
    with open('inference/m4singer/m4singer/m4singer_pinyin2ph.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split('|') if x.strip() != '']
            pinyin2phs[elements[0]] = elements[1]
    return pinyin2phs