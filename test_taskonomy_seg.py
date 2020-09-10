from test_models2 import *

models_root = "/local/rcs/ECCV/Taskonomy/selected"

# Baselines
DEBUG = False

map_dict = {
    'segmentsemantic'   : 's',
    'edge_texture'      : 'e',
    'depth_zbuffer'     : 'd',
    'autoencoder'       : 'A',
    'edge_occlusion'    : 'E',
    'depth_euclidean'   : 'D',
    'normal'            : 'n',
    'principal_curvature': 'p',
    'reshading'         : 'r',
    'keypoints2d'       : 'k',
    'keypoints3d'       : 'K'
}

# # Adversarial Attack for Single task
#
# test_task = 'segmentsemantic'
# seed=0
# results = test_one_checkpoint(
#                 os.path.join(models_root, "trainset_{}_testset_{}_lambda_0.0_seed_{}/savecheckpoint/checkpoint_150.pth.tar".format(map_dict[test_task], map_dict[test_task], seed)),
#                 "resnet18",
#                 task_set_whole=[test_task],
#                 test_task_set=[test_task], test_batch_size=256, steps=50, debug=DEBUG,
#                 epsilon=4, step_size=1)

# Adversarial Attack for Multitask
# adversarial attack for Segmentation under Multitask:
test_tasks = ['segmentsemantic']

for test_task in test_tasks:
    aux_tasks = ['segmentsemantic', 'depth_zbuffer', 'edge_texture', 'normal', 'reshading', 'keypoints2d', 'keypoints3d', 'depth_euclidean', 'autoencoder', 'edge_occlusion', 'principal_curvature']
    lambda_list = [0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.1, 0.01, 0.1, 0.01]
    aux_tasks.remove(test_task)

    results_all = {}

    for aux_task, l in zip(aux_tasks, lambda_list):
        # results_all['lambda_{}'.format(l)] = {}
        results = test_one_checkpoint(
                    os.path.join(models_root, "trainset_{}{}_testset_{}_lambda_{}/savecheckpoint/checkpoint_150.pth.tar".format(map_dict[test_task], map_dict[aux_task], map_dict[test_task], l)),
                    "resnet18",
                    task_set_whole=[test_task, aux_task],
                    test_task_set=[test_task], test_batch_size=128, steps=50, debug=DEBUG,
                    epsilon=4, step_size=1)

        if test_task == 'segmentsemantic':
            results_all[aux_task] = results['advacc'][test_task]['iou']
        else:
            results_all[aux_task] = results['advacc'][test_task]

    print(results_all)

