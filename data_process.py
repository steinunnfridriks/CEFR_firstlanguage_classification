import csv
import glob
from progress.bar import IncrementalBar
import sys

def make_csv():
    """
    A function to get the texts from IGC_texts_unreviewed into a CSV format to be used with 
    level_analyser.py and/or logisticregression.py
    """

    folders = ["IGC_texts_unreviewed/A2", "IGC_texts_unreviewed/B1", "IGC_texts_unreviewed/B2", "IGC_texts_unreviewed/C1", "IGC_texts_unreviewed/C2"]

    with open("IGC_texts.csv", 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IGCsource", "text", "cefrlevel"])            
    
        for folder in folders:
            level = folder.split("/")[1]
            files = glob.glob(f'{folder}/*.txt', recursive=True)

            filebar = IncrementalBar('Inntaksskjöl lesin', max = len(files))
            for file in files:
                with open(file, 'r', encoding='utf-8') as content:
                    row = []
                    count = 0
                    for line in content:
                        if count == 0:
                            row.append(line.rstrip())
                        elif count == 1:
                            pass
                        else:
                            row.append(line.strip())
                        count += 1
                    row.append(level)
                writer.writerow(row)        

                filebar.next()
                sys.stdout.flush()
            filebar.finish()            

make_csv()

