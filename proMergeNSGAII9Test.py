import os, shutil, random


base_path = os.path.join(os.path.abspath(r'./'), r'datSet')
source_path = [
    ['ZDT/ZDT123/datX1d', 'ZDT/ZDT4/datXd5', 'ZDT/ZDT6/datXd',
     'FON/datXd', 'KUR/datXd', 'POL/datXd2', 'SCH/datXd1'],
    ['ZDT/ZDT123/datY1d', 'ZDT/ZDT4/datYd5', 'ZDT/ZDT6/datYd',
     'FON/datYd', 'KUR/datYd', 'POL/datYd2', 'SCH/datYd1'],
    ['ZDT/ZDT123/datZ1d', 'ZDT/ZDT4/datZd5', 'ZDT/ZDT6/datZd',
     'FON/datZd', 'KUR/datZd', 'POL/datZd2', 'SCH/datZd1'],
    ]
dist_path = [
    'NSGA_II_9_test/datX', 'NSGA_II_9_test/datY', 'NSGA_II_9_test/datZ'
]
len_dir = []
special_file = ['imgX_ZDT', 'imgY_ZDT', 'imgZ_ZDT',
                'imgX_SCH', 'imgY_SCH', 'imgZ_SCH',
                'imgX_POL', 'imgY_POL', 'imgZ_POL',
                'imgX_KUR', 'imgY_KUR', 'imgZ_KUR',
                'imgX_FON', 'imgY_FON', 'imgZ_FON',
                ]
for a in source_path[0]:
    path = os.path.join(base_path, a)
    print(len(os.listdir(path)))
    len_dir.append(len(os.listdir(path)))

new_name_index = 0
for i in range(7):
    
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
