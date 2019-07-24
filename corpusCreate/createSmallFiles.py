lines_per_file = 5000
smallfile = None
i = 1
with open('output1_es.txt' , encoding="utf8") as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = '{}.txt'.format(i)
            smallfile = open(small_filename, "w", encoding="utf8")
            i = i + 1
        smallfile.write(line)
    if smallfile:
        smallfile.close()
