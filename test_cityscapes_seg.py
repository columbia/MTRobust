DEBUG=False
import socket
import os

if socket.gethostname() == 'deep':
    root_path ="/mnt/md0/2019Fall/SegSaveLog/cityscape/multi-task-small/drn-22/visualize_model/"
elif socket.gethostname() == 'hulk':
    root_path = "/local/rcs/ECCV/Cityscape/Houdini/lambda0.01"
    # root_path = "/local/rcs/mcz/MultiTask/cityscape/multi-task-small/drn-22/visualize_model/"
single = ["adv_s-only-lam0.01", "adv_s-sd-lam0.01", "adv_s-sA-lam0.01", "adv_s-sdA-lam0.01"]

# single = ["trainset_s_testset_s_lambda_0_2", "trainset_sd_testset_s_lambda_0.001",
#           "trainset_sA_testset_s_lambda_0.1", "trainset_sdA_testset_s_lambda_0.001"]
    # list_name = [["segmentsemantic"] for i in range(9)]

list_name = [["segmentsemantic"], ["segmentsemantic", "depth_zbuffer"],
             ["segmentsemantic", "autoencoder"], ["segmentsemantic", "depth_zbuffer", "autoencoder"]]



model_path_list = []
for i, each in enumerate(single):
    model_path_list.append(os.path.join(root_path, each))
print("root", root_path)

from test_models2 import *
batch_size=16

# PGD Attack
num_steps = 20 # example for attack steps, can set to other number of steps
for ii in range(4):

    print("\n\n\nindividually PGD Attack, num steps {}".format(num_steps))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["segmentsemantic"],
        test_batch_size=batch_size,
        steps= num_steps,
        debug=DEBUG,
        epsilon=4,
        step_size=1,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=False,
        use_houdini=False)

# MIM Attack
for ii in range(4):
    print("\n\n\nindividually MIM Attack, num steps {}".format(num_steps))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["segmentsemantic"],
        test_batch_size=batch_size,
        steps= num_steps,
        debug=DEBUG,
        epsilon=4,
        step_size=1,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=True,
        use_houdini=False)

# Houdini Attack
for ii in range(4):
    print("\n\n\nindividually Houdini Attack, num steps {}".format(num_steps))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["segmentsemantic"],
        test_batch_size=batch_size,
        steps= num_steps,
        debug=DEBUG,
        epsilon=4,
        step_size=1,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=True,
        use_houdini=True)

