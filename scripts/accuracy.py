import sys

def read_from_file(filename):
    with open(filename, "r") as f:
        gt = []
        for line in f:
            if line.startswith("\t"):
                continue
            if "\t" not in line:
                continue
            l = line.split("\t")
            query_id = int(l[0])
            doc_id = int(l[1])
            if query_id == len(gt):
                gt.append(set())
            gt[query_id].add(doc_id)
    return gt

gt = read_from_file(sys.argv[1])
r = read_from_file(sys.argv[2])

counter = 0
tot = 0
for s1, s2 in zip(gt,r):
    counter += len(s1 & s2)
    tot += len(s1)

recall = counter/tot*100
print(f"\nRecall: {counter/tot*100:.2f}\n")
print(f"{counter/tot*100:.2f}", file=sys.stderr)


