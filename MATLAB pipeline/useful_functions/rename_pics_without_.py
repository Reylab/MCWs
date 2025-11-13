import os

def rename_pics(pics_folder):
    pics = os.listdir(pics_folder)
    for pic in pics:
        # Skip if file has underscore in the name
        if '_' in pic:
            continue
        # Split the name of the file into the name and the extension
        name, ext = os.path.splitext(pic)
        # Split the name of the file into the name and the number
        name, number = name.rstrip('0123456789-'), name[len(name.rstrip('0123456789')):]
        # Rename the file with an underscore between the name and the number
        os.rename(os.path.join(pics_folder, pic), os.path.join(pics_folder, name + '_' + number + ext))

def add_concept_num_to_end(pics_folder, concept_num=1):
    pics = os.listdir(pics_folder)
    for pic in pics:
        # Split the name of the file into the name and the extension
        name, ext = os.path.splitext(pic)
        # Rename the file with an underscore between the name and the number
        os.rename(os.path.join(pics_folder, pic), os.path.join(pics_folder, name + '_' + str(concept_num) + ext))

if __name__ == '__main__':
    rename_pics('/home/user/share/experimental_files/pics/pics_space/seman_test/')