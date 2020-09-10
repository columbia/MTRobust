import socket


# Set root path to the models trained.
if socket.gethostname() == 'deep':
    root_path ="/mnt/md0/2019Fall/SegSaveLog/cityscape/multi-task-small/drn-22-d-related/visualize_model/"
elif socket.gethostname() == 'hulk':
    root_path = "/local/rcs/mcz/MultiTask/cityscape/multi-task-small/drn-22-d-related/visualize_model/"


single = ["trainset_d_testset_d_lambda_0.1_seed_227_lrs_140_200",
"trainset_sd_testset_d_lambda_0.1_seed_42_lrs_140_200",
          "trainset_dA_testset_d_lambda_0.1_seed_42_lrs_140_200",
"trainset_sdA_testset_d_lambda_0.01_seed_42_lrs_140_200"]

    # list_name = [["segmentsemantic"] for i in range(9)]

list_name = [["depth_zbuffer"], ["segmentsemantic", "depth_zbuffer"],
             ["depth_zbuffer", "autoencoder"], ["segmentsemantic", "depth_zbuffer", "autoencoder"]]


model_path_list = []
for i, each in enumerate(single):
    model_path_list.append(root_path + each)
print("root", root_path)

from test_models2 import *

batch_size=12
DEBUG=False
EPSILON=4

STEP=1

#  PGD attack
for ii in range(4):
    step_num = 100  # number of steps for PGD attack
    print("\n\n\nindividually PGD step={}".format(step_num))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["depth_zbuffer"],
        test_batch_size=batch_size,
        steps=step_num,
        debug=DEBUG,
        epsilon=EPSILON,
        step_size=STEP,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True)

# MIM attack
for ii in range(4):
    step_num = 100 # number of steps for PGD attack
    print("\n\n\nindividually MIM step={}".format(step_num))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["depth_zbuffer"],
        test_batch_size=batch_size,
        steps=step_num,
        debug=DEBUG,
        epsilon=EPSILON,
        step_size=STEP,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=True)