all_labels = {
    'Faces': [],
    'Faces_easy': [],
    'Leopards': [288, 289],
    'Motorbikes': [670],
    'accordion': [401],
    'airplanes': [404, 405, 895],
    'anchor': [],
    'ant': [310],
    'barrel': [427, 412, 756],
    'bass': [],
    'beaver': [337],
    'binocular': [447],
    'bonsai': [],
    'brain': [109], # brain coral!
    'brontosaurus': [],
    'buddha': [],
    'butterfly': [322, 323, 324, 325, 326],
    'camera': [732, 759],
    'cannon': [471],
    'car_side': [705, 751, 817, 436],
    'ceiling_fan': [545],
    'cellphone': [487],
    'chair': [559, 765, 423],
    'chandelier': [],
    'cougar_body': [286],
    'cougar_face': [286],
    'crab': [118, 119, 120, 121, 125],
    'crayfish': [123, 124],
    'crocodile': [49, 50],
    'crocodile_head': [49, 50],
    'cup': [968, 647],
    'dalmatian': [251],
    'dollar_bill': [],
    'dolphin': [],
    'dragonfly': [319],
    'electric_guitar': [546],
    'elephant': [385, 386],
    'emu': [],
    'euphonium': [],
    'ewer': [725],
    'ferry': [],
    'flamingo': [130],
    'flamingo_head': [130],
    'garfield': [],
    'gerenuk': [],
    'gramophone': [],
    'grand_piano': [579],
    'hawksbill': [],
    'headphone': [],
    'hedgehog': [334],
    'helicopter': [],
    'ibis': [],
    'inline_skate': [],
    'joshua_tree': [],
    'kangaroo': [104],
    'ketch': [],
    'lamp': [846],
    'laptop': [620],
    'llama': [355],
    'lobster': [122, 123],
    'lotus': [],
    'mandolin': [],
    'mayfly': [],
    'menorah': [],
    'metronome': [],
    'minaret': [],
    'nautilus': [117],
    'octopus': [],
    'okapi': [],
    'pagoda': [],
    'panda': [],
    'pigeon': [],
    'pizza': [963],
    'platypus': [103],
    'pyramid': [],
    'revolver': [763],
    'rhino': [],
    'rooster': [],
    'saxophone': [776],
    'schooner': [780],
    'scissors': [],
    'scorpion': [71],
    'sea_horse': [],
    'snoopy': [],
    'soccer_ball': [805],
    'stapler': [],
    'starfish': [327],
    'stegosaurus': [],
    'stop_sign': [919],
    'strawberry': [949],
    'sunflower': [],
    'tick': [78],
    'trilobite': [69],
    'umbrella': [879],
    'watch': [531, 826],
    'water_lilly': [],
    'wheelchair': [],
    'wild_cat': [],
    'windsor_chair': [],
    'wrench': [],
    'yin_yang': []
}

labels = {
    'Leopards': [288, 289],
    'Motorbikes': [670],
    'accordion': [401],
    'airplanes': [404, 405, 895],
    'ant': [310],
    'barrel': [427, 412, 756],
    'beaver': [337],
    'binocular': [447],
    'brain': [109],
    'butterfly': [322, 323, 324, 325, 326],
    'camera': [732, 759],
    'cannon': [471],
    'car_side': [705, 751, 817, 436],
    'ceiling_fan': [545],
    'cellphone': [487],
    'chair': [559, 765, 423],
    'cougar_body': [286],
    'cougar_face': [286],
    'crab': [118, 119, 120, 121, 125],
    'crayfish': [123, 124],
    'crocodile': [49, 50],
    'crocodile_head': [49, 50],
    'cup': [968, 647],
    'dalmatian': [251],
    'dragonfly': [319],
    'electric_guitar': [546],
    'elephant': [385, 386],
    'ewer': [725],
    'flamingo': [130],
    'flamingo_head': [130],
    'grand_piano': [579],
    'hedgehog': [334],
    'kangaroo': [104],
    'lamp': [846],
    'laptop': [620],
    'llama': [355],
    'lobster': [122, 123],
    'nautilus': [117],
    'pizza': [963],
    'platypus': [103],
    'revolver': [763],
    'saxophone': [776],
    'schooner': [780],
    'scorpion': [71],
    'soccer_ball': [805],
    'starfish': [327],
    'stop_sign': [919],
    'strawberry': [949],
    'tick': [78],
    'trilobite': [69],
    'umbrella': [879],
    'watch': [531, 826]
}

numeric_labels = {
    'Leopards': 36,
    'Motorbikes': 21,
    'accordion': 6,
    'airplanes': 42,
    'ant': 28,
    'barrel': 49,
    'beaver': 30,
    'binocular': 32,
    'brain': 23,
    'butterfly': 38,
    'camera': 9,
    'cannon': 15,
    'car_side': 31,
    'ceiling_fan': 17,
    'cellphone': 46,
    'chair': 5,
    'cougar_body': 11,
    'cougar_face': 26,
    'crab': 10,
    'crayfish': 50,
    'crocodile': 3,
    'crocodile_head': 19,
    'cup': 47,
    'dalmatian': 45,
    'dragonfly': 13,
    'electric_guitar': 44,
    'elephant': 35,
    'ewer': 8,
    'flamingo': 22,
    'flamingo_head': 39,
    'grand_piano': 16,
    'hedgehog': 51,
    'kangaroo': 43,
    'lamp': 34,
    'laptop': 1,
    'llama': 18,
    'lobster': 29,
    'nautilus': 27,
    'pizza': 37,
    'platypus': 48,
    'revolver': 33,
    'saxophone': 25,
    'schooner': 12,
    'scorpion': 7,
    'soccer_ball': 40,
    'starfish': 14,
    'stop_sign': 0,
    'strawberry': 24,
    'tick': 20,
    'trilobite': 41,
    'umbrella': 4,
    'watch': 2
}



def get_label_name(label):
    for k, v in labels.iteritems():
        if label in v:
            return k
    return None

def get_small_label_name(label):
    for k, v in numeric_labels.iteritems():
        if int(label) == int(v):
            return k
    return None
