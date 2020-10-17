# Script to calculate average change per layer for deep_weight_logs


def calcDiff(a, b):
    toReturn = 0
    for i0 in range(0, len(a)):
        toReturn += abs(float(a[i0]) - float(b[i0]))
    return toReturn

def calc_avg_change_per_layer(filename):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    text = text[0:-2] # Last element is | so just remove it
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("\n", " ")
    layerVals = text.split("|")
    avgChangeByLayer = [0]*11
    for i0 in range(0, len(layerVals)-1):
        fromLayer = layerVals[i0].split(" ")
        fromLayer = [ele for ele in fromLayer if ele != ""]
        toLayer = layerVals[i0+1].split(" ")
        toLayer = [ele for ele in toLayer if ele != ""]
        for i1 in range(0, 11):
            avgChangeByLayer[i1] += calcDiff(fromLayer[i1*9:i1*9+9], toLayer[i1*9:i1*9+9])/9
    avgChangeByLayer = [ele/len(layerVals) for ele in avgChangeByLayer]
    return avgChangeByLayer

f = open("avgChangeByLayer.txt", "w")
for ele in ["deep_weight_logs/01leakyrelu.txt", "deep_weight_logs/001leakyrelu.txt", "deep_weight_logs/0001leakyrelu.txt",
            "deep_weight_logs/00001leakyrelu.txt", "deep_weight_logs/000001leakyrelu.txt", "deep_weight_logs/01sigmoid.txt",
            "deep_weight_logs/001sigmoid.txt", "deep_weight_logs/0001sigmoid.txt",
            "deep_weight_logs/00001sigmoid.txt", "deep_weight_logs/000001sigmoid.txt"]:
    avgChange = calc_avg_change_per_layer(ele)
    f.writelines([ele + ": ", str(avgChange), "\n"])
f.close()
