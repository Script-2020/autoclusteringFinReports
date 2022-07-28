import matplotlib.pyplot as plt
import pandas as pd 
import squarify
import os

class ChartDrawer:
   
    @staticmethod
    def drawTreeMap(clustersList, dicClusters, total_documents,output_dir):
        countCluster = [len(item[1]['documents']) for item in clustersList]
        df = pd.array(countCluster)

        sizes = countCluster
        colors = [plt.cm.Spectral(countCluster[i] / max(countCluster)) for i in range(len(countCluster))]
        plt.figure(figsize=(8, 12), dpi=80)
        squarify.plot(sizes=sizes, label=sizes, color=colors, alpha=.8)
        plt.title('DISTRIBUTION OF DOCUMENTS PER CLUSTERS\n' + '%s clusters, %s clustered documents (%s' % (
        str(len(countCluster)), str(sum(countCluster)), str(int(sum(countCluster) / total_documents * 100))) + '%)')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir,"clusteredDocs.jpg"))
        plt.show()
