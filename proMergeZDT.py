import os, shutil, random


base_path = os.path.join(os.path.abspath(r'./'), r'datSet/ZDT')
source_path = [
    ['ZDT123/datX1d', 'ZDT4/datXd5', 'ZDT6/datXd'],
    ['ZDT123/datY1d', 'ZDT4/datYd5', 'ZDT6/datYd'],
    ['ZDT123/datZ1d', 'ZDT4/datZd5', 'ZDT6/datZd'],
    ]
dist_path = ['ZDT12346/datXd1', 'ZDT12346/datYd1', 'ZDT12346/datZd1']
len_dir = []
special_file = ['imgX_ZDT', 'imgY_ZDT', 'imgZ_ZDT']
for a in source_path[0]:
    path = os.path.join(base_path, a)
    print(len(os.listdir(path)))
    len_dir.append(len(os.listdir(path)))

new_name_index = 0
for i in range(3):
    
    source_X_path = os.path.join(base_path, source_path[0][i])
    source_Y_path = os.path.join(base_path, source_path[1][i])
    source_Z_path = os.path.join(base_path, source_path[2][i])

    dist_X_path = os.path.join(base_path, dist_path[0])
    dist_Y_path = os.path.join(base_path, dist_path[1])
    dist_Z_path = os.path.join(base_path, dist_path[2])

    pre_X_names = os.listdir(source_X_path)
    
    for pre_X_name in pre_X_names:
        
        if pre_X_name.endswith('.bmp'):
            pre_Y_name = 'imgY' + pre_X_name[4:]
            pre_Z_name = 'imgZ' + pre_X_name[4:]

            if pre_X_name[: 8] in special_file:
                new_X_name = pre_X_name
                new_Y_name = pre_Y_name
                new_Z_name = pre_Z_name
            else:
                new_X_name = 'imgX_{}.bmp'.format(new_name_index)
                new_Y_name = 'imgY_{}.bmp'.format(new_name_index)
                new_Z_name = 'imgZ_{}.bmp'.format(new_name_index)
                
                new_name_index += 1

            shutil.copyfile(
                os.path.join(source_X_path, pre_X_name),
                os.path.join(dist_X_path, new_X_name)
                )
            shutil.copyfile(
                os.path.join(source_Y_path, pre_Y_name),
                os.path.join(dist_Y_path, new_Y_name)
                )
            shutil.copyfile(
                os.path.join(source_Z_path, pre_Z_name),
                os.path.join(dist_Z_path, new_Z_name)
                )

print(new_name_index)
