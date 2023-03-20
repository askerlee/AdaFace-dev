subjects = []
initwords = []

# read all lines from sublist.txt
with open('sublist.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        subject, initword = line.split(',')
        subjects.append(subject)
        initwords.append(initword)

print("\t".join(subjects))
print("\t".join(initwords))
