# ic_thresholds = [
#     1, # super potent
#     10, # potent
#     100, # very promising
#     500, # very promising
#     1000, # Mild
#     # Rest -> inactive
# ]

ic_thresholds = [
    10, # super potent
    100, # potent
    1000, # very promising
    5000,
    # Rest -> inactive
]

def quantize_ic50(x):
    broke = False
    for class_id, thresh in enumerate(ic_thresholds):
        if x < thresh:
            broke = True
            break
    if not broke:
        class_id = len(ic_thresholds)
    return class_id
